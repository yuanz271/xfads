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


class AddBias(Module):
    bias: Array
    out_features: Union[int, Literal["scalar"]] = field(static=True)

    def __init__(
        self,
        out_features: Union[int, Literal["scalar"]],
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `in_features`: The input size. The input to the layer should be a vector of
            shape `(in_features,)`
        - `out_features`: The output size. The output from the layer will be a vector
            of shape `(out_features,)`.
        - `use_bias`: Whether to add on a bias as well.
        - `dtype`: The dtype to use for the weight and the bias in this layer.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Note that `in_features` also supports the string `"scalar"` as a special value.
        In this case the input to the layer should be of shape `()`.

        Likewise `out_features` can also be a string `"scalar"`, in which case the
        output from the layer will have shape `()`.
        """
        out_features_ = 1 if out_features == "scalar" else out_features
        lim = 1 / math.sqrt(out_features_)

        bshape = (out_features_,)
        self.bias = jrandom.uniform(key, bshape, minval=-lim, maxval=lim)

        self.out_features = out_features

    @jax.named_scope("xfads.nn.AddBias")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        x = x + self.bias
        if self.out_features == "scalar":
            assert jnp.shape(x) == (1,)
            x = jnp.squeeze(x)
        return x


class VariantBiasLinear(Module):
    biases: Array
    linear: Module

    def __init__(self, state_dim, observation_dim, n_biases, *, key, norm_readout: bool = False):
        wkey, bkey = jrandom.split(key, 2)

        self.linear = Linear(state_dim, observation_dim, key=wkey, use_bias=False)
        if norm_readout:
            self.linear = WeightNorm(self.linear)
        
        lim = 1 / math.sqrt(state_dim)
        self.biases = jrandom.uniform(bkey, (n_biases, observation_dim), dtype=self.linear.weight.dtype, minval=-lim, maxval=lim)

        # bkeys = jrandom.split(bkey, n_biases)
        # self.add_bias = [AddBias(observation_dim, key=bkey).__call__ for bkey in bkeys]

    
    def __call__(self, idx, x):
        x = self.linear(x)
        return x + self.biases[idx]
        # return jax.lax.switch(idx, self.add_bias, x)
