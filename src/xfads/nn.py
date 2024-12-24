from functools import partial
import math
from typing import Any, Callable, Literal, Optional, Sequence, TypeVar, Union

import jax
from jax import nn as jnn, random as jrandom, numpy as jnp
from jaxtyping import PRNGKeyArray, Array, Scalar
from equinox import nn as enn, Module, field, tree_at
from equinox.nn import Linear


EPS = 1e-6


def make_mlp(in_size, out_size, width, depth, *, key: PRNGKeyArray, activation: Callable=jnn.tanh):
    keys = jrandom.split(key, depth + 2)
    layers = [enn.Linear(in_size, width, key=keys[0]), enn.Lambda(activation)]
    for i in range(depth):
        layers.append(enn.Linear(width, width, key=keys[i + 1]))
        layers.append(enn.Lambda(activation))
    layers.append(enn.Linear(width, out_size, key=keys[-1]))
    return enn.Sequential(layers)


def softplus_inverse(x):
    return x + jnp.log(-jnp.expm1(-x))


def _norm_except_axis(v: Array, norm: Callable[[Array], Scalar], axis: Optional[int]):
    if axis is None:
        return norm(v)
    else:
        return jax.vmap(norm, in_axes=axis, out_axes=axis)(v)
    

class WeightNorm(Module):
    layer: Linear
    weight_name: str = field(static=True)
    axis: Optional[int] = field(static=True)
    _norm: Callable[[Array], Scalar]

    def __init__(
        self,
        layer: Linear,
        weight_name: str = "weight",
        axis: Optional[int] = None,
    ):
        """
        :param layer: The layer to wrap. equinox.nn.Linear
        :param weight_name: The name of the layer's parameter (a JAX array) to apply weight normalisation to.
        :param axis: The norm is computed across every axis except this one. If `None`, compute across every axis.
        """
        self.layer = layer
        self.weight_name = weight_name
        self.axis = axis

        self._norm = partial(
            _norm_except_axis,
            norm=partial(jnp.linalg.norm, keepdims=True),
            axis=axis,
        )
    
    @property
    def weight(self) -> Array:
        w = getattr(self.layer, self.weight_name)
        w = w / (self._norm(w) + EPS)

        return w
    
    @property
    def bias(self) -> Array:
        return self.layer.bias

    @jax.named_scope("xfads.nn.WeightNorm")
    def __call__(self, x: Array) -> Array:
        """
        :param x: JAX Array
        """
        weight = self.weight
        layer = tree_at(
            lambda layer: getattr(layer, self.weight_name), self.layer, weight
        )
        return layer(x)


class VariantBiasLinear(Module):
    biases: Array = field(static=True)
    linear: Module

    def __init__(self, state_dim, observation_dim, n_biases, biases, *, key, norm_readout: bool = False):
        wkey, bkey = jrandom.split(key, 2)

        self.linear = Linear(state_dim, observation_dim, key=wkey, use_bias=True)
        if norm_readout:
            self.linear = WeightNorm(self.linear)
        
        if biases == "none":
            lim = 1 / math.sqrt(state_dim)
            self.biases = jrandom.uniform(bkey, (n_biases, observation_dim), dtype=self.linear.weight.dtype, minval=-lim, maxval=lim)
        else:
            self.biases = biases
    
    def __call__(self, idx, x):
        x = self.linear(x)
        return x + self.biases[idx]
        # return jax.lax.switch(idx, self.add_bias, x)
