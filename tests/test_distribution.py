from jax import numpy as jnp, random as jrandom
import chex
import tensorflow_probability.substrates.jax.distributions as tfp

from xfads.distribution import MVN, DiagMVN


def test_mvn(dimensions):
    state_dim, input_dim, observation_dim = dimensions
    m1 = jnp.ones(state_dim)
    cov1 = jnp.eye(state_dim)
    m2 = jnp.zeros(state_dim)
    cov2 = jnp.eye(state_dim)
    
    moment1 = MVN.canon_to_moment(m1, cov1)
    moment2 = MVN.canon_to_moment(m2, cov2)
    kl = MVN.kl(moment1, moment2)
    chex.assert_tree_all_finite(kl)


def test_diagmvn(dimensions):
    state_dim, input_dim, observation_dim = dimensions
    m1 = jnp.ones(state_dim)
    cov1 = jnp.ones(state_dim)
    m2 = jnp.zeros(state_dim)
    cov2 = jnp.ones(state_dim) * 2
    
    moment = DiagMVN.canon_to_moment(m1, cov1)
    nat1 = DiagMVN.moment_to_natural(moment)
    moment1 = DiagMVN.natural_to_moment(nat1)
    
    chex.assert_trees_all_close(moment, moment1)

    moment2 = DiagMVN.canon_to_moment(m2, cov2)
    kl = DiagMVN.kl(moment1, moment2)
    chex.assert_tree_all_finite(kl)


def test_reparameterization(dimensions):
    state_dim, input_dim, observation_dim = dimensions
    m1 = jnp.ones(state_dim)
    cov1 = jnp.eye(state_dim)
    assert tfp.MultivariateNormalFullCovariance(m1, cov1).reparameterization_type == tfp.FULLY_REPARAMETERIZED
