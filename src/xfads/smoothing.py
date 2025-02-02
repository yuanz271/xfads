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
from .distribution import MVN


EPS = 1e-5


@dataclass
class Hyperparam:
    approx: Type[MVN]
    state_dim: int
    input_dim: int
    observation_dim: int
    covariate_dim: int
    mc_size: int
    regular: float = 0.01


def smooth(
    t: Array,
    y: Array,
    u: Array,
    key: PRNGKeyArray,
    dynamics: Module,
    statenoise: Module,
    likelihood: Likelihood,
    obs_to_update: Module,
    back_encoder: Module,
    hyperparam: Hyperparam,
) -> tuple[Array, Array]:
    approx = hyperparam.approx
    natural_to_moment = jax.vmap(approx.natural_to_moment)
    
    # y_hat = jax.vmap(likelihood.predict)(t)
    # residual = jnp.log(y + EPS) - jnp.log(y_hat + EPS)
    # residual = y

    update_obs = jax.vmap(lambda x: approx.constrain_natural(obs_to_update(x)))(y)
    nature_prior_1 = approx.prior_natural(hyperparam.state_dim)
    
    nature_f_1 = nature_prior_1 + update_obs[0]
    moment_f_1 = approx.natural_to_moment(nature_f_1)

    expected_moment = partial(sample_expected_moment, forward=dynamics, noise=statenoise, approx=approx, mc_size=hyperparam.mc_size)

    def forward(carry, obs):
        key, nature_f_tm1 = carry
        subkey, key_t = jrandom.split(key, 2)
        update_obs_t, u = obs
        moment_s_tm1 = approx.natural_to_moment(nature_f_tm1)
        moment_p_t = expected_moment(key_t, moment_s_tm1, u)
        nature_p_t = approx.moment_to_natural(moment_p_t)
        nature_f_t = nature_p_t + update_obs_t
        return (subkey, nature_f_t), (moment_p_t, nature_p_t, nature_f_t)

    def backward(carry, z):
        key, nature_s_tp1 = carry
        subkey, key_t = jrandom.split(key, 2)
        info_y_t, nature_f_t = z
        update = approx.constrain_natural(back_encoder(jnp.concatenate((info_y_t, nature_s_tp1))))
        nature_s_t = nature_f_t + update
        return (subkey, update), nature_s_t

    key, forward_key = jrandom.split(key, 2)
    _, (moment_p, nature_p, nature_f) = scan(
        forward, init=(forward_key, nature_f_1), xs=(update_obs[1:], u[1:])
    )  #
    
    moment_p = jnp.vstack((moment_f_1, moment_p))
    nature_f = jnp.vstack((nature_f_1, nature_f))
    moment_s = natural_to_moment(nature_f)

    ## Backward
    nature_s_T = nature_f[-1]   
    key, backward_key = jrandom.split(key, 2)
    _, nature_s = scan(
        backward, init=(backward_key, nature_s_T), xs=(y[:-1], nature_f[:-1]), reverse=True
    )  # reverse both xs and the output
    nature_s = jnp.vstack((nature_s, nature_s_T))
    moment_s = natural_to_moment(nature_s)
    
    # expectation should be under smoothing distribution
    keys = jrandom.split(key, jnp.size(moment_s, 0))
    moment_p = jax.vmap(expected_moment)(keys, moment_s, u)
    moment_p = jnp.vstack((moment_s[0], moment_p[:-1]))

    return moment_s, moment_p
