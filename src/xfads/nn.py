from functools import partial
from typing import Callable, Optional
from jaxtyping import PRNGKeyArray, Array, Scalar
import jax
from jax import nn as jnn, random as jrandom, numpy as jnp
from equinox import nn as enn, Module, field, tree_at
from equinox.nn import Linear


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
        axis: Optional[int] = 0,
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

    def weight(self) -> Array:
        w = getattr(self.layer, self.weight_name)
        weight = w / self._norm(w)
        
        return weight
    
    @jax.named_scope("xfads.nn.WeightNorm")
    def __call__(self, x: Array) -> Array:
        """
        :param x: JAX Array
        """
        weight = self.weight()
        layer = tree_at(lambda layer: getattr(layer, self.weight_name), self.layer, weight)
        return layer(x)
    