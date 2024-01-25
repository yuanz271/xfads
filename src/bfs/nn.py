from typing import Callable
from jaxtyping import PRNGKeyArray
from jax import nn as jnn, random as jrandom, numpy as jnp
from equinox import nn as enn


def make_mlp(in_size, out_size, hidden_size, n_layers, *, key: PRNGKeyArray, activation: Callable=jnn.selu):
    keys = jrandom.split(key, n_layers + 2)
    layers = [enn.Linear(in_size, hidden_size, key=keys[0]), enn.Lambda(activation)]
    for i in range(n_layers):
        layers.append(enn.Linear(hidden_size, hidden_size, key=keys[i + 1]))
        layers.append(enn.Lambda(jnn.silu))
    layers.append(enn.Linear(hidden_size, out_size, key=keys[-1]))
    return enn.Sequential(layers)


def softplus_inverse(x):
    return x + jnp.log(-jnp.expm1(-x))
