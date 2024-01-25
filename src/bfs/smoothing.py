from dataclasses import dataclass
from typing import Type
from equinox import Module
from jaxtyping import Array, PRNGKeyArray
import jax
from jax import numpy as jnp, nn as jnn, random as jrandom
from jax.lax import scan

from .vi import Likelihood
from .dynamics import sample_expected_moment
from .distribution import ExponentialFamily
from .nn import make_mlp


def get_obs_encoder(
    state_dim: int,
    observation_dim: int,
    hidden_size: int,
    n_layers: int,
    *,
    key: PRNGKeyArray,
):
    return make_mlp(
        observation_dim, approx.moment_size(state_dim), hidden_size, n_layers, key=key
    )


def get_back_encoder(
    state_dim: int, hidden_size: int, n_layers: int, approx: Type[ExponentialFamily], *, key: PRNGKeyArray
):
    return make_mlp(
        approx.moment_size(state_dim) * 2, approx.moment_size(state_dim), hidden_size, n_layers, key=key
    )


def constrained_natural(unconstrained):
    nat1, nat2 = jnp.split(unconstrained, 2)
    nat2 = -jnn.softplus(nat2)
    return jnp.concatenate((nat1, nat2), axis=-1)


@dataclass
class Hyperparam:
    approx: Type[ExponentialFamily]
    state_dim: int
    input_dim: int
    observation_dim: int
    mc_size: int


def smooth(
    y: Array,
    u: Array,
    key: PRNGKeyArray,
    dynamics: Module,
    statenoise: Module,
    likelihood: Likelihood,
    obs_encoder: Module,
    back_encoder: Module,
    hyperparam: Hyperparam,
) -> tuple[Array, Array]:
    approx = hyperparam.approx
    nature_f_1 = constrained_natural(obs_encoder(y[0]))

    def forward(carry, obs):
        key, nature_f_tm1 = carry
        key, key_t = jrandom.split(key, 2)
        y, u = obs
        moment_s_tm1 = approx.natural_to_moment(nature_f_tm1)
        moment_p_t = sample_expected_moment(
            key_t, dynamics, statenoise, moment_s_tm1, u, approx, hyperparam.mc_size
        )
        nature_p_t = approx.moment_to_natural(moment_p_t)
        update = constrained_natural(obs_encoder(y))
        nature_f_t = nature_p_t + update
        return (key, nature_f_t), (nature_p_t, nature_f_t)

    def backward(carry, z):
        key, nature_s_tp1 = carry
        key, key_t = jrandom.split(key, 2)
        nature_f_t, u = z

        update = constrained_natural(back_encoder(nature_s_tp1))
        nature_s_t = nature_f_t + update
        moment_s_t = approx.natural_to_moment(nature_s_t)
        moment_p_tp1 = sample_expected_moment(
            key_t, dynamics, statenoise, moment_s_t, u, approx, hyperparam.mc_size
        )
        return (key, nature_s_t), (moment_p_tp1, nature_s_t)

    key, key_f = jrandom.split(key, 2)
    _, (nature_p, nature_f) = scan(
        forward, init=(key_f, nature_f_1), xs=(y[1:], u[1:])
    )  #
    nature_p = jnp.vstack((nature_f_1, nature_p))
    nature_f = jnp.vstack((nature_f_1, nature_f))

    nature_s_T = nature_f[-1]
    key, key_b = jrandom.split(key, 2)
    _, (moment_p, nature_s) = scan(
        backward, init=(key_b, nature_s_T), xs=(nature_f[:-1], u[:-1]), reverse=True
    )  # reverse both xs and the output
    nature_s = jnp.vstack((nature_s, nature_s_T))
    moment_s = jax.vmap(approx.natural_to_moment)(nature_s)
    moment_s_1 = moment_s[0]
    moment_p = jnp.vstack((moment_s_1, moment_p))
    nature_p = jax.vmap(approx.moment_to_natural)(moment_p)

    return moment_s, moment_p
