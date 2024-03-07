from jax import numpy as jnp, random as jrnd
from equinox import nn as enn
import chex

from xfads.distribution import MVN
from xfads.vi import PoissonLik, elbo


def test_poisson(dimensions):
    key = jrnd.PRNGKey(0)
    state_dim, input_dim, observation_dim = dimensions

    y = jrnd.poisson(key, jnp.ones(observation_dim))
    chex.assert_shape(y, (observation_dim,))
    readout = enn.Linear(state_dim, observation_dim, key=key)
    lik = PoissonLik(readout)
    
    m = jnp.ones(state_dim)
    cov = jnp.eye(state_dim)
    moment = MVN.canon_to_moment(m, cov)
    chex.assert_shape(moment, (state_dim * state_dim + state_dim,))
    
    ell = lik.eloglik(key, moment, y, mc_size=10)
    chex.assert_shape(ell, y.shape)
    chex.assert_tree_all_finite(ell)
    

def test_elbo(dimensions):
    key = jrnd.PRNGKey(0)
    state_dim, input_dim, observation_dim = dimensions

    m1 = jnp.ones(state_dim)
    cov1 = jnp.eye(state_dim)
    m2 = jnp.zeros(state_dim)
    cov2 = jnp.eye(state_dim)
    
    moment1 = MVN.canon_to_moment(m1, cov1)
    moment2 = MVN.canon_to_moment(m2, cov2)

    y = jrnd.poisson(key, jnp.ones(observation_dim))
    readout = enn.Linear(state_dim, observation_dim, key=key)
    lik = PoissonLik(readout)

    l = elbo(key, moment1, moment2, y, lik.eloglik, MVN, mc_size=10)
    chex.assert_tree_all_finite(l)
