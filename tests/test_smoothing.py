from jax import numpy as jnp, random as jrandom
from equinox import nn as enn
import chex
import jax
import pytest

from xfads.distribution import DiagMVN
from xfads.dynamics import GaussianStateNoise, Nonlinear
from xfads.smoothing import get_back_encoder, get_obs_encoder, smooth
from xfads.smoothing import Hyperparam
from xfads.vi import DiagGaussainLik


def test_smooth(dimensions, capsys):
    key = jrandom.PRNGKey(0)
    key, dyn_key, obs_key, back_key = jrandom.split(key, 4)

    state_dim, input_dim, observation_dim = dimensions
    hidden_size: int = 2
    n_layers: int = 2
    T: int = 100

    f = Nonlinear(state_dim, input_dim, key=dyn_key, kwargs={'width': hidden_size, 'depth': n_layers})

    obs_encoder = get_obs_encoder(state_dim, observation_dim, hidden_size, n_layers, DiagMVN, key=obs_key)
    back_encoder = get_back_encoder(state_dim, hidden_size, n_layers, DiagMVN, key=back_key)
    
    key, ykey, ukey, rkey, skey = jrandom.split(key, 5)

    y = jrandom.normal(ykey, shape=(T, observation_dim))
    u = jrandom.normal(ukey, shape=(T, input_dim))

    hyperparam = Hyperparam(DiagMVN, state_dim, input_dim, observation_dim, mc_size=10)
    statenoise = GaussianStateNoise(jnp.ones(state_dim))
    linear_readout = enn.Linear(state_dim, observation_dim, key=rkey)
    likelihood = DiagGaussainLik(cov=jnp.ones(observation_dim), readout=linear_readout)
    
    with capsys.disabled():
        moment_s, moment_p = smooth(y, u, skey, dynamics=f, statenoise=statenoise, likelihood=likelihood, obs_encoder=obs_encoder, back_encoder=back_encoder, hyperparam=hyperparam)
