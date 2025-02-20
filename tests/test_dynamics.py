from jax import numpy as jnp, random as jrnd
import chex
from xfads.distributions import MVN

from xfads.dynamics import Nonlinear, predict_moment, sample_expected_moment, DiagNoise


def test_nonlinear(spec):
    key = jrnd.key(0)
    state_dim = spec['state_dim']
    input_dim = spec['input_dim']
    
    f = Nonlinear(state_dim, input_dim, key=key, kwargs=spec['dyn_spec'])
    z = jrnd.normal(key, (state_dim,))
    u = jrnd.normal(key, (input_dim,))
    fz = f(z, u)
    chex.assert_equal_shape((z, fz))


def test_predict_moment(spec):
    key = jrnd.key(0)
    state_dim = spec['state_dim']
    input_dim = spec['input_dim']

    f = Nonlinear(state_dim, input_dim, key=key, kwargs=spec['dyn_spec'])
    noise = DiagNoise(jnp.eye(state_dim))

    z = jrnd.normal(key, (state_dim,))
    u = jrnd.normal(key, (input_dim,))

    moment = predict_moment(z, u, f, noise, MVN)
    chex.assert_shape(moment, (MVN.moment_size(state_dim),))


def test_sample_expected_moment(spec):
    key = jrnd.key(0)
    state_dim = spec['state_dim']
    input_dim = spec['input_dim']

    f = Nonlinear(state_dim, input_dim, key=key, kwargs=spec['dyn_spec'])
    noise = DiagNoise(jnp.eye(state_dim))

    z = jrnd.normal(key, (state_dim,))
    u = jrnd.normal(key, (input_dim,))
    
    moment = predict_moment(z, u, f, noise, MVN)
    moment = sample_expected_moment(key, moment, u, f, noise, MVN, 10)
    chex.assert_shape(moment, (MVN.moment_size(state_dim),))
