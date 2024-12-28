from typing import Any, Callable, ClassVar
import jax
from jax import numpy as jnp, nn as jnn, random as jrandom
from jaxtyping import Array, PRNGKeyArray
from equinox import Module, nn as enn
import tensorflow_probability.substrates.jax.distributions as tfp
import chex

from .nn import make_mlp


class ImplicitDiagMVN(Module):
    """d grad log p(y | z) / d mu
    Diagonal
    """
    mlp: Module

    def __init__(self, state_dim: int, observation_dim: int, width: int, depth: int, *, key: PRNGKeyArray):
        self.mlp = make_mlp(observation_dim, state_dim*2, width, depth, key=key)

    def __call__(self, y: Array) -> tuple[Array, Array]:
        output = self.mlp(y)
        nat1, nat2 = jnp.split(output, 2, axis=-1)
        return nat1, jnn.softplus(nat2)


class BackwardDiagMVN(Module):
    cell: Module
    output_layer: Module
    h0: Array

    def __init__(self, state_dim: int, observation_dim: int, width: int, *, key: PRNGKeyArray):
        rnn_key, output_key, hkey = jrandom.split(key, 3)
        self.h0 = jrandom.normal(hkey, shape=(width,))
        self.cell = enn.GRUCell(observation_dim, width, key=rnn_key)
        self.output_layer = enn.Linear(width, state_dim*2, key=output_key)

    def __call__(self, y: Array) -> tuple[Array, Array]:
        h0 = jnn.tanh(self.h0)
        
        def scan_fn(h, x):
            h = self.cell(x, h)
            return h, h
 
        _, hs = jax.lax.scan(scan_fn, h0, y)
        output = jax.vmap(self.output_layer)(hs)
        nat1, nat2 = jnp.split(output, 2, axis=-1)
        return nat1, jnn.softplus(nat2)


class PseudoObservation(Module):
    cell: Module
    output_layer: Module
    h0: Array

    def __init__(self, input_size: int, hidden_size: int, output_size: int, *, key: PRNGKeyArray):
        cell_key, hidden_key, output_key = jrandom.split(key, 3)
        self.h0 = jrandom.normal(hidden_key, shape=(hidden_key,))
        self.cell = enn.GRUCell(input_size, hidden_size, key=cell_key)
        self.output_layer = enn.Linear(hidden_size, output_size, key=output_key)

    def __call__(self, y: Array) -> tuple[Array, Array]:
        h0 = jnn.tanh(self.h0)
        
        def scan_fn(h, x):
            h = self.cell(x, h)
            return h, h
 
        _, hs = jax.lax.scan(scan_fn, h0, y, reverse=True)
        output = jax.vmap(self.output_layer)(hs)
        return output
