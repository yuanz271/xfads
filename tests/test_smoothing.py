from jax import numpy as jnp, random as jrandom

import xfads
from xfads.dynamics import DiagNoise, Nonlinear
from xfads.distributions import DiagMVN, FullMVN, LoRaMVN
from xfads.smoothing import smooth
from xfads.smoothing import Hyperparam


def test_smooth(spec, capsys):
    key, dyn_key, enc_key = jrandom.split(jrandom.key(0), 3)

    state_dim = spec['state_dim']
    observation_dim = spec['observation_dim']
    input_dim = spec['input_dim']
    depth = spec['depth']
    width = spec['width']

    approx = LoRaMVN

    T: int = 100

    f = Nonlinear(state_dim, input_dim, width, depth, key=dyn_key)

    obs_encoder, back_encoder = approx.get_encoders(observation_dim, state_dim, width, depth, enc_key)
    
    key, ykey, ukey, rkey, skey = jrandom.split(key, 5)

    y = jrandom.normal(ykey, shape=(T, observation_dim))
    u = jrandom.normal(ukey, shape=(T, input_dim))

    hyperparam = Hyperparam(approx, state_dim, input_dim, observation_dim, covariate_dim=spec['covariate_dim'], mc_size=spec['mc_size'])
    statenoise = DiagNoise(jnp.ones(state_dim))
    
    with capsys.disabled():
        print(approx)
        moment_s, moment_p = smooth(y, u, key=skey, dynamics=f, statenoise=statenoise, obs_to_update=obs_encoder, back_encoder=back_encoder, hyperparam=hyperparam, likelihood=None)
