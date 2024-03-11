from typing import Callable
from jaxtyping import PRNGKeyArray
from jax import nn as jnn, random as jrandom, numpy as jnp
from equinox import nn as enn


def make_mlp(in_size, out_size, hidden_size, n_layers, *, key: PRNGKeyArray, activation: Callable=jnn.silu):
    keys = jrandom.split(key, n_layers + 1)
    layers = []
    for i in range(n_layers):
        layers.append(enn.Linear(in_size, hidden_size, key=keys[i]))
        layers.append(enn.Lambda(activation))
        in_size = hidden_size
    layers.append(enn.Linear(in_size, out_size, key=keys[-1]))
    return enn.Sequential(layers)


def softplus_inverse(x):
    return x + jnp.log(-jnp.expm1(-x))
