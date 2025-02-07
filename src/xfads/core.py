
from functools import partial
from jaxtyping import Array, PRNGKeyArray, PyTree
import jax
from jax import numpy as jnp, nn as jnn, random as jrandom
from jax.lax import scan

from .smoothing import Hyperparam
from .vi import Likelihood
from .dynamics import sample_expected_moment
from .distribution import MVN


def filter(
    modules: PyTree,
    key: PRNGKeyArray,
    t: Array,
    y: Array,
    u: Array,
    hyperparam: Hyperparam,
) -> tuple[Array, Array]:
    dynamics, likelihood, obs_to_update, back_encoder = modules

    approx = hyperparam.approx
    natural_to_moment = jax.vmap(approx.natural_to_moment)

    # y_res = jax.vmap(likelihood.residual)(t, y)

    update_obs = jax.vmap(lambda x: approx.constrain_natural(obs_to_update(x)))(y)
    nature_prior_1 = approx.prior_natural(hyperparam.state_dim)  # TODO: trainable prior?

    nature_f_1 = nature_prior_1 + update_obs[0]
    moment_f_1 = approx.natural_to_moment(nature_f_1)

    expected_moment = partial(
        sample_expected_moment,
        forward=dynamics,
        noise=dynamics,
        approx=approx,
        mc_size=hyperparam.mc_size,
    )

    def forward(carry, obs):
        key, nature_f_tm1 = carry
        subkey, mckey = jrandom.split(key, 2)
        update_obs_t, u = obs
        moment_s_tm1 = approx.natural_to_moment(nature_f_tm1)
        moment_p_t = expected_moment(mckey, moment_s_tm1, u)
        nature_p_t = approx.moment_to_natural(moment_p_t)
        nature_f_t = nature_p_t + update_obs_t
        return (subkey, nature_f_t), (moment_p_t, nature_f_t)

    key, forward_key = jrandom.split(key, 2)
    _, (moment_p, nature_f) = scan(
        forward, init=(forward_key, nature_f_1), xs=(update_obs[1:], u[1:])
    )  #

    moment_p = jnp.vstack((moment_f_1, moment_p))
    nature_f = jnp.vstack((nature_f_1, nature_f))
    moment_f = natural_to_moment(nature_f)

    return nature_f, moment_f, moment_p


def smooth(
    modules: PyTree,
    key: PRNGKeyArray,
    t: Array,
    y: Array,
    u: Array,
    hyperparam: Hyperparam,
):
    dynamics, likelihood, obs_to_update, back_encoder = modules
    approx = hyperparam.approx
    natural_to_moment = jax.vmap(approx.natural_to_moment)
    
    info_obs = jax.vmap(obs_to_update)(y)
    update_obs = jax.vmap(approx.constrain_natural)(info_obs)

    nature_prior_1 = approx.prior_natural(hyperparam.state_dim)
    
    nature_f_1 = nature_prior_1 + update_obs[0]
    moment_f_1 = approx.natural_to_moment(nature_f_1)

    expected_moment = partial(sample_expected_moment, forward=dynamics, noise=dynamics, approx=approx, mc_size=hyperparam.mc_size)

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
        backward, init=(backward_key, nature_s_T), xs=(info_obs[:-1], nature_f[:-1]), reverse=True
    )  # reverse both xs and the output
    nature_s = jnp.vstack((nature_s, nature_s_T))
    moment_s = natural_to_moment(nature_s)
    
    # expectation should be under smoothing distribution
    keys = jrandom.split(key, jnp.size(moment_s, 0))
    moment_p = jax.vmap(expected_moment)(keys, moment_s, u)
    moment_p = jnp.vstack((moment_s[0], moment_p[:-1]))

    return nature_s, moment_s, moment_p


def bismooth(
    model,
    key: PRNGKeyArray,
    t: Array,
    y: Array,
    u: Array,
    hyperparam: Hyperparam,
):
    """
    Bidirectional filtering
    Parameterize inverse dynamics
    q(z[t]|y[1:T]) = q(z[y]|y[1:t])q(z[t]|y[t+1:T])/p(z[t])
    P[t] = Pf[t] + Pb[t] - P0
    Pm = Pfmf + Pbmb
    See Dowling23 Eq.(21-23)
    """
    # forward_dynamics, likelihood, obs_to_update, backward_dynamics = modules

    approx = hyperparam.approx

    natural_to_moment = jax.vmap(approx.natural_to_moment)
    expected_moment_forward = partial(sample_expected_moment, f=model.forward, noise=model.forward, approx=approx, mc_size=hyperparam.mc_size)
    expected_moment_backward = partial(sample_expected_moment, f=model.backward, noise=model.backward, approx=approx, mc_size=hyperparam.mc_size)    
    
    update_by_obs = jax.vmap(approx.constrain_natural)(jax.vmap(model.obs_encoder)(y))

    nature_prior = approx.prior_natural(hyperparam.state_dim)
    
    nature_f_1 = nature_prior + update_by_obs[0]

    def ff(carry, obs, expected_moment):
        key, nature_f_tm1 = carry
        key_tp1, key_t = jrandom.split(key, 2)
        update_obs_t, u = obs
        moment_f_tm1 = approx.natural_to_moment(nature_f_tm1)
        moment_p_t = expected_moment(key_t, moment_f_tm1, u)
        nature_p_t = approx.moment_to_natural(moment_p_t)
        nature_f_t = nature_p_t + update_obs_t
        return (key_tp1, nature_f_t), (moment_p_t, nature_p_t, nature_f_t)

    # def bf(carry, obs):
    #     key, nature_f_tm1 = carry
    #     subkey, key_t = jrandom.split(key, 2)
    #     update_obs_t, u = obs
    #     moment_f_tm1 = approx.natural_to_moment(nature_f_tm1)
    #     moment_p_t = expected_moment_backward(key_t, moment_f_tm1, u)
    #     nature_p_t = approx.moment_to_natural(moment_p_t)
    #     nature_f_t = nature_p_t + update_obs_t
    #     return (subkey, nature_f_t), (moment_p_t, nature_p_t, nature_f_t)

    # Forward
    key, forward_key = jrandom.split(key, 2)
    _, (_, _, nature_f) = scan(
        partial(ff, expected_moment=expected_moment_forward), init=(forward_key, nature_f_1), xs=(update_by_obs[1:], u[:-1])  # t = 2 ... T+1
    )
    nature_f = jnp.vstack((nature_f_1, nature_f))  # 1...T

    ## Backward
    key, subkey = jrandom.split(key, 2)
    (_, nature_b_Tp1), _ = ff((subkey, nature_f[-1]), (jnp.zeros_like(nature_prior), u[-1]), expected_moment_forward)

    key, backward_key = jrandom.split(key, 2)
    _, (_, nature_p_b, _) = scan(
        partial(ff, expected_moment=expected_moment_backward), init=(backward_key, nature_b_Tp1), xs=(update_by_obs, u), reverse=True
    )

    nature_s = nature_f + nature_p_b - jnp.expand_dims(nature_prior, axis=0)
    moment_s = natural_to_moment(nature_s)
    
    # expectation should be under smoothing distribution
    keys = jrandom.split(key, jnp.size(moment_s, 0))
    moment_p = jax.vmap(expected_moment_forward)(keys, moment_s, u)
    moment_p = jnp.vstack((moment_s[0], moment_p[:-1]))

    return nature_s, moment_s, moment_p


def smooth_pseudo(
    modules: PyTree,
    key: PRNGKeyArray,
    t: Array,
    y: Array,
    u: Array,
    hyperparam: Hyperparam,
):
    dynamics, likelihood, obs_to_update, pseudo_encoder = modules
    approx = hyperparam.approx
    natural_to_moment = jax.vmap(approx.natural_to_moment)

    nature_prior_1 = approx.prior_natural(hyperparam.state_dim)  # TODO: trainable prior?
    
    info_obs = jax.vmap(obs_to_update)(y)

    update_obs = jax.vmap(approx.constrain_natural)(info_obs)
    
    pseudo_obs = update_obs
    pseudo_obs = pseudo_obs + jax.vmap(approx.constrain_natural)(pseudo_encoder(jnp.concatenate((y, info_obs), axis=-1)))
    
    # pseudo_obs = jax.vmap(approx.constrain_natural)(pseudo_encoder(y)) 
    nature_f_1 = nature_prior_1 + pseudo_obs[0]
    moment_f_1 = approx.natural_to_moment(nature_f_1)

    expected_moment = partial(
        sample_expected_moment,
        forward=dynamics,
        noise=dynamics,
        approx=approx,
        mc_size=hyperparam.mc_size,
    )
    
    def forward(carry, obs):
        key, nature_f_tm1 = carry
        subkey, mckey = jrandom.split(key, 2)
        pseudo_obs_t, u = obs
        moment_s_tm1 = approx.natural_to_moment(nature_f_tm1)
        moment_p_t = expected_moment(mckey, moment_s_tm1, u)
        nature_p_t = approx.moment_to_natural(moment_p_t)
        nature_f_t = nature_p_t + pseudo_obs_t
        return (subkey, nature_f_t), (moment_p_t, nature_f_t)

    key, forward_key = jrandom.split(key, 2)
    _, (moment_p, nature_f) = scan(
        forward, init=(forward_key, nature_f_1), xs=(pseudo_obs[1:], u[1:])
    )  #

    moment_p = jnp.vstack((moment_f_1, moment_p))
    nature_f = jnp.vstack((nature_f_1, nature_f))
    moment_f = natural_to_moment(nature_f)

    return nature_f, moment_f, moment_p
