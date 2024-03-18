from tempfile import TemporaryDirectory
from jax import numpy as jnp, random as jrandom
from equinox import nn as enn
import chex
import equinox as eqx

from xfads.distribution import DiagMVN
from xfads.dynamics import GaussianStateNoise, Nonlinear
from xfads.smoothing import get_back_encoder, get_obs_encoder
from xfads.smoother import XFADS, load_model, save_model
from xfads.smoothing import Hyperparam
from xfads.trainer import Opt, make_batch_elbo, make_batch_smoother, train_em
from xfads.vi import DiagGaussainLik


def test_batch_smoother(dimensions, capsys):
    key = jrandom.PRNGKey(0)
    key, dyn_key, obs_key, back_key = jrandom.split(key, 4)

    state_dim, input_dim, observation_dim = dimensions
    hidden_size: int = 2
    n_layers: int = 2
    N: int = 10
    T: int = 100

    f = Nonlinear(state_dim, input_dim, key=dyn_key, kwargs={'width': hidden_size, 'depth': n_layers})

    obs_encoder = get_obs_encoder(
        state_dim, observation_dim, hidden_size, n_layers, DiagMVN, key=obs_key
    )
    back_encoder = get_back_encoder(state_dim, hidden_size, n_layers, DiagMVN, key=back_key)

    key, ykey, ukey, rkey, skey = jrandom.split(key, 5)

    y = jrandom.normal(ykey, shape=(N, T, observation_dim))
    u = jrandom.normal(ukey, shape=(N, T, input_dim))

    hyperparam = Hyperparam(DiagMVN, state_dim, input_dim, observation_dim, mc_size=10)
    statenoise = GaussianStateNoise(jnp.ones(state_dim))
    linear_readout = enn.Linear(state_dim, observation_dim, key=rkey)
    likelihood = DiagGaussainLik(cov=jnp.ones(observation_dim), readout=linear_readout)

    smooth = make_batch_smoother(
        f, statenoise, likelihood, obs_encoder, back_encoder, hyperparam
    )
    moment_s, moment_p = smooth(y, u, key)
    chex.assert_equal_size((moment_s, moment_p))


def test_batch_elbo(dimensions, capsys):
    key = jrandom.PRNGKey(0)
    key, rkey, ykey, ykey, ukey, skey, pkey = jrandom.split(key, 7)

    state_dim, input_dim, observation_dim = dimensions
    N: int = 10
    T: int = 100

    linear_readout = enn.Linear(state_dim, observation_dim, key=rkey)
    likelihood = DiagGaussainLik(cov=jnp.ones(observation_dim), readout=linear_readout)

    elbo = make_batch_elbo(likelihood.eloglik, DiagMVN, mc_size=10)
    ys = jrandom.normal(ykey, (N, T, observation_dim))
    us = jrandom.normal(ukey, (N, T, input_dim))
    moment_s = jrandom.uniform(skey, (N, T, state_dim * 2))
    moment_p = jrandom.uniform(pkey, (N, T, state_dim * 2))
    
    with capsys.disabled():
        L = elbo(key, moment_s, moment_p, ys)


def test_xfads(dimensions, capsys):
    state_dim, input_dim, observation_dim = dimensions
    hidden_size: int = 2
    n_layers: int = 2
    N: int = 10
    T: int = 100
    key = jrandom.PRNGKey(0)
    
    model_spec = dict(observation_dim=observation_dim,
        state_dim=state_dim,
        input_dim=input_dim,
        approx="DiagMVN",
        dyn_mod="Nonlinear",
        mc_size=10,
        random_state=0,
        emission_noise=1.,
        state_noise=1.,
        enc_kwargs={'width': hidden_size, 'depth': n_layers},
        dyn_kwargs={'width': hidden_size, 'depth': n_layers},
    )
    xfads = XFADS(**model_spec)

    key, ykey, ukey, fkey, skey = jrandom.split(key, 5)

    y = jrandom.normal(ykey, shape=(N, T, observation_dim))
    u = jrandom.normal(ukey, shape=(N, T, input_dim))
    
    xfads.fit((y, u), key=fkey, mode="joint")
    
    with TemporaryDirectory() as tmpdir:
        save_model(f"{tmpdir}/model.eqx", model_spec, xfads)
        loaded_model = load_model(f"{tmpdir}/model.eqx")
    
    chex.assert_trees_all_equal(eqx.filter(xfads.dynamics, eqx.is_array), eqx.filter(loaded_model.dynamics, eqx.is_array))
