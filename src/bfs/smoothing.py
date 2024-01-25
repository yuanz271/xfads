from dataclasses import dataclass
from functools import partial
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
    approx: Type[ExponentialFamily],
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
    nat1, nat2 = jnp.split(unconstrained, 2, axis=-1)
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
    nature_y = constrained_natural(jax.vmap(obs_encoder)(y))
    nature_f_1 = nature_y[0]
    moment_f_1 = approx.natural_to_moment(nature_f_1)

    expected_moment = partial(sample_expected_moment, forward=dynamics, noise=statenoise, approx=approx, mc_size=hyperparam.mc_size)
    natural_to_moment = jax.vmap(approx.natural_to_moment)

    def forward(carry, obs):
        key, nature_f_tm1 = carry
        subkey, key_t = jrandom.split(key, 2)
        nature_y_t, u = obs
        moment_s_tm1 = approx.natural_to_moment(nature_f_tm1)
        moment_p_t = expected_moment(key_t, moment_s_tm1, u)
        nature_p_t = approx.moment_to_natural(moment_p_t)
        nature_f_t = nature_p_t + nature_y_t
        return (subkey, nature_f_t), (moment_p_t, nature_p_t, nature_f_t)

    def backward(carry, z):
        key, nature_s_tp1 = carry
        subkey, key_t = jrandom.split(key, 2)
        nature_y_t, nature_f_t = z

        update = constrained_natural(back_encoder(jnp.concatenate((nature_y_t, nature_s_tp1), axis=-1)))
        nature_s_t = nature_f_t + update
        return (subkey, update), nature_s_t

    key, forward_key = jrandom.split(key, 2)
    _, (moment_p, nature_p, nature_f) = scan(
        forward, init=(forward_key, nature_f_1), xs=(nature_y[1:], u[1:])
    )  #
    
    moment_p = jnp.vstack((moment_f_1, moment_p))
    nature_f = jnp.vstack((nature_f_1, nature_f))
    moment_s = natural_to_moment(nature_f)

    ## Backward
    nature_s_T = nature_f[-1]   
    update_s_T = jnp.zeros_like(nature_s_T)
    key, backward_key = jrandom.split(key, 2)
    _, nature_s = scan(
        backward, init=(backward_key, update_s_T), xs=(nature_y[:-1], nature_f[:-1]), reverse=True
    )  # reverse both xs and the output
    nature_s = jnp.vstack((nature_s, nature_s_T))
    moment_s = natural_to_moment(nature_s)
    moment_s_1 = moment_s[0]
    
    keys = jrandom.split(key, jnp.size(moment_s, 0))
    moment_p = jax.vmap(expected_moment)(keys, moment_s, u)
    moment_p = jnp.vstack((moment_s_1, moment_p[:-1]))

    return moment_s, moment_p
