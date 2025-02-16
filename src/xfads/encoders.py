import math
from typing import Callable, Type
import jax
from jax import numpy as jnp, nn as jnn, random as jrandom
from jaxtyping import Array, Float
import equinox as eqx

from .nn import make_mlp


class AlphaEncoder(eqx.Module):
    approx: Type = eqx.field(static=True)
    layer: eqx.Module

    def __init__(self, state_dim, observation_dim, depth, width, approx, *, key):
        self.approx = approx
        
        self.layer = make_mlp(
            observation_dim, approx.param_size(state_dim), width, depth, key=key
        )
    
    def __call__(self, y):
        return self.layer(y)
    

class BetaEncoder(eqx.Module):
    approx: Type = eqx.field(static=True)
    h0: Array
    cell: eqx.Module
    output: eqx.Module
    # dropout: eqx.nn.Dropout

    def __init__(self, state_dim, depth, width, approx, *, key):
        self.approx = approx
        
        param_size = approx.param_size(state_dim)

        key, subkey = jrandom.split(key)
        lim = 1 / math.sqrt(width)
        self.h0 = jrandom.uniform(subkey, (width,), minval=-lim, maxval=lim)

        key, subkey = jrandom.split(key)
        self.cell = eqx.nn.GRUCell(param_size, width, key=subkey)

        key, subkey = jrandom.split(key)
        self.output = eqx.nn.Linear(width, param_size, key=subkey)
    
    def __call__(self, a: Float[Array, "t h"], *, key):
        """
        :param a: natural form observation information
        """
        def step(h, inp):
            h = self.cell(inp, h)
            return h, h
        
        _, hs = jax.lax.scan(step, init=self.h0, xs=a, reverse=True)

        ab = jax.vmap(self.output)(hs)

        return ab
