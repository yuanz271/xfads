from jax import numpy as jnp, random as jrandom

import xfads
from xfads.dynamics import DiagMVNStateNoise, Nonlinear
from xfads.smoothing import get_back_encoder, get_neural_to_state, smooth
from xfads.smoothing import Hyperparam


def test_smooth(spec, capsys):
    key = jrandom.key(0)
    key, dyn_key, obs_key, back_key = jrandom.split(key, 4)

    state_dim = spec['state_dim']
    neural_dim = spec['neural_dim']
    input_dim = spec['input_dim']

    approx = getattr(xfads.distribution, spec["approx"])

    T: int = 100

    f = Nonlinear(state_dim, input_dim, key=dyn_key, kwargs=spec['dyn_spec'])

    obs_encoder = get_neural_to_state(state_dim, neural_dim, approx, key=obs_key, **spec['enc_spec'])
    back_encoder = get_back_encoder(state_dim, approx, key=back_key, **spec['enc_spec'])
    
    key, ykey, ukey, rkey, skey = jrandom.split(key, 5)

    y = jrandom.normal(ykey, shape=(T, neural_dim))
    u = jrandom.normal(ukey, shape=(T, input_dim))

    hyperparam = Hyperparam(approx, state_dim, input_dim, neural_dim, mc_size=spec['mc_size'], covariate_dim=spec['covariate_dim'])
    state_noise = DiagMVNStateNoise(jnp.ones(state_dim))
    
    with capsys.disabled():
        print(approx)
        moment_s, moment_p = smooth(y, u, skey, dynamics=f, state_noise=state_noise, obs_encoder=obs_encoder, back_encoder=back_encoder, hyperparam=hyperparam)
