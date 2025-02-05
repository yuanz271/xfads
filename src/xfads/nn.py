from functools import partial
import math
from mimetypes import init
from typing import Callable, Optional

import jax
from jax import nn as jnn, random as jrandom, numpy as jnp
from jaxtyping import PRNGKeyArray, Array, Scalar
import equinox as eqx
from equinox import nn as enn, Module
from equinox.nn import Linear


EPS = 1e-6


def make_mlp(in_size, out_size, width, depth, *, key: PRNGKeyArray, activation: Callable=jnn.swish):
    key, layer_key = jrandom.split(key)
    layers = [enn.Linear(in_size, width, key=layer_key), enn.Lambda(activation)]
    for i in range(depth - 1):
        key, layer_key = jrandom.split(key) 
        layers.append(enn.Linear(width, width, key=layer_key))
        layers.append(enn.Lambda(activation))
    key, layer_key = jrandom.split(key)
    layers.append(enn.Linear(width, out_size, key=layer_key))
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
    weight_name: str = eqx.field(static=True)
    axis: Optional[int] = eqx.field(static=True)
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
        layer = eqx.ree_at(
            lambda layer: getattr(layer, self.weight_name), self.layer, weight
        )
        return layer(x)


class VariantBiasLinear(Module):
    biases: Array = eqx.field(static=False)
    linear: Module

    def __init__(self, state_dim, observation_dim, n_biases, biases, *, key, norm_readout: bool = False):
        wkey, bkey = jrandom.split(key, 2)

        self.linear = Linear(state_dim, observation_dim, key=wkey, use_bias=False)
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
    
    def set_static(self, static=True) -> None:
        self.__dataclass_fields__['biases'].metadata = {'static': static}


def gauss_rbf(x, c, s):
    return jnp.exp(-jnp.sum(jnp.square((x - c) * s)))


class RBFN(Module):
    centers: Array = eqx.field(static=True)
    scale: Array 
    readout: Module

    def __init__(self, input_size, output_size, network_size, *, key, normalized: bool = False):
        key, ckey=jrandom.split(key)
        self.centers = jrandom.uniform(ckey, shape=(network_size, input_size), minval=-1, maxval=1)
        self.scale = jnp.ones(input_size)
        self.readout = Linear(network_size, output_size, key=key)

    def __call__(self, x):
        kernels = jax.vmap(gauss_rbf, in_axes=(None, 0, None))(x, self.centers, jnn.softplus(self.scale))
        return self.readout(kernels)
