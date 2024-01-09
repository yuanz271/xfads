from typing import Any, Callable, ClassVar
import jax
from jax import numpy as jnp, nn as jnn, random as jrnd
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

    def __init__(self, state_dim: int, observation_dim: int, hidden_size: int, n_layers: int, *, key: PRNGKeyArray):
        self.mlp = make_mlp(observation_dim, state_dim*2, hidden_size, n_layers, key=key)

    def __call__(self, y: Array) -> tuple[Array, Array]:
        output = self.mlp(y)
        nat1, nat2 = jnp.split(output, 2)
        return nat1, jnn.softplus(nat2)


class BackwardDiagMVN(Module):
    cell: Module
    output_layer: Module
    h0: Array

    def __init__(self, state_dim: int, observation_dim: int, hidden_size: int, *, key: PRNGKeyArray):
        rnn_key, output_key, hkey = jrnd.split(key, 3)
        self.h0 = jrnd.normal(hkey, shape=(hidden_size,))
        self.cell = enn.GRUCell(observation_dim, hidden_size, key=rnn_key)
        self.output_layer = enn.Linear(hidden_size, state_dim*2, key=output_key)

    def __call__(self, y: Array) -> tuple[Array, Array]:
        h0 = jnn.tanh(self.h0)
        
        def scan_fn(h, x):
            h = self.cell(x, h)
            return h, h
 
        _, hs = jax.lax.scan(scan_fn, h0, y)
        output = jax.vmap(self.output_layer)(hs)
        nat1, nat2 = jnp.split(output, 2, axis=-1)
        return nat1, jnn.softplus(nat2)
