from jax import numpy as jnp, random as jrnd
import chex
import pytest
from xfads.distribution import MVN

from xfads.dynamics import GaussianStateNoise, Nonlinear, predict_moment, sample_expected_moment


def test_nonlinear(dimensions):
    key = jrnd.PRNGKey(0)
    state_dim, input_dim, observation_dim = dimensions
    hidden_size = 2
    n_layers = 2
    f = Nonlinear(state_dim, input_dim, key=key, kwargs={'width': hidden_size, 'depth': n_layers})
    z = jrnd.normal(key, (2,))
    u = jrnd.normal(key, (2,))
    fz = f(z, u)
    chex.assert_equal_shape((z, fz))


def test_predict_moment(dimensions):
    key = jrnd.PRNGKey(0)
    state_dim, input_dim, observation_dim = dimensions
    hidden_size = 2
    n_layers = 2
    f = Nonlinear(state_dim, input_dim, key=key, kwargs={'width': hidden_size, 'depth': n_layers})
    noise = GaussianStateNoise(jnp.eye(state_dim))

    z = jrnd.normal(key, (state_dim,))
    u = jrnd.normal(key, (input_dim,))

    moment = predict_moment(z, u, f, noise, MVN)
    chex.assert_shape(moment, (state_dim + state_dim * state_dim,))


def test_sample_expected_moment(dimensions):
    key = jrnd.PRNGKey(0)
    state_dim, input_dim, observation_dim = dimensions
    hidden_size: int = 2
    n_layers: int = 2
    f = Nonlinear(state_dim, input_dim, key=key, kwargs={'width': hidden_size, 'depth': n_layers})
    noise = GaussianStateNoise(jnp.eye(state_dim))

    z = jrnd.normal(key, (state_dim,))
    u = jrnd.normal(key, (input_dim,))
    
    moment = predict_moment(z, u, f, noise, MVN)
    moment = sample_expected_moment(key, moment, u, f, noise, MVN, 10)
    chex.assert_shape(moment, (state_dim + state_dim * state_dim,))
