from typing import Callable
from sklearn.base import TransformerMixin
from jaxtyping import Array, Float, PRNGKeyArray
import jax
from jax import numpy as jnp, nn as jnn, random as jrnd
from equinox import Module, nn as enn


def make_mlp(in_size, out_size, hidden_size, n_layers, *, key: PRNGKeyArray, activation: Callable=jnn.selu):
    keys = jrnd.split(key, n_layers + 2)
    layers = [enn.Linear(in_size, hidden_size, key=keys[0]), enn.Lambda(activation)]
    for i in range(n_layers):
        layers.append(enn.Linear(hidden_size, hidden_size, key=keys[i + 1]))
        layers.append(enn.Lambda(jnn.silu))
    layers.append(enn.Linear(hidden_size, out_size, key=keys[i + 1]))
    return enn.Sequential(layers)
