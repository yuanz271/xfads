from functools import partial
import math
from typing import Callable, Optional

import numpy as np
import jax
from jax import nn as jnn, random as jrandom, numpy as jnp
from jaxtyping import ArrayLike, PRNGKeyArray, Array, Scalar
import equinox as eqx
from equinox import nn as enn, Module


_MIN_NORM = 1e-6
MAX_EXP = 5.
EPS = np.finfo(np.float32).eps

def make_mlp(
    in_size,
    out_size,
    width,
    depth,
    *,
    key: PRNGKeyArray,
    activation: Callable = jnn.swish,
    final_bias: bool = True,
    final_activation: Callable | None = None,
    dropout: float | None = None,
) -> Module:
    key, layer_key = jrandom.split(key)
    layers = [enn.Linear(in_size, width, key=layer_key), enn.Lambda(activation)]
    for i in range(depth - 1):
        key, layer_key = jrandom.split(key)
        layers.append(enn.Linear(width, width, key=layer_key))
        layers.append(enn.Lambda(activation))
        if dropout is not None:
            layers.append(enn.Dropout(dropout))
    key, layer_key = jrandom.split(key)
    layers.append(enn.Linear(width, out_size, key=layer_key, use_bias=final_bias))
    if final_activation is not None:
        layers.append(enn.Lambda(activation))
    if dropout is not None:
        layers.append(enn.Dropout(dropout))
    
    return enn.Sequential(layers)


# def softplus(x):
#     """A numerical safe implementation"""
#     return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0)


# def softplus_inverse(x: ArrayLike):
    # return jnp.log(jnp.maximum(jnp.expm1(x), 1e-6))


def constrain_positive(x):
    # x0 = MAX_EXP
    # is_too_large = x > x0
    # expx0 = jnp.exp(x0)
    # clipped = jnp.where(is_too_large, x0, x)
    # expx = jnp.exp(clipped)
    # taylor = expx0 + expx0 * (x - x0)
    # return jnp.where(is_too_large, taylor, expx)
    return jnp.square(x) + EPS


def unconstrain_positive(x):
    # return jnp.log(x + EPS)
    return jnp.sqrt(x)


def softplus_inverse(x: ArrayLike):
    # From TF
    # We begin by deriving a more numerically stable softplus_inverse:
    # x = softplus(y) = Log[1 + exp{y}], (which means x > 0).
    # ==> exp{x} = 1 + exp{y}                                (1)
    # ==> y = Log[exp{x} - 1]                                (2)
    #       = Log[(exp{x} - 1) / exp{x}] + Log[exp{x}]
    #       = Log[(1 - exp{-x}) / 1] + Log[exp{x}]
    #       = Log[1 - exp{-x}] + x                           (3)
    # (2) is the "obvious" inverse, but (3) is more stable than (2) for large x.
    # For small x (e.g. x = 1e-10), (3) will become -inf since 1 - exp{-x} will
    # be zero. To fix this, we use 1 - exp{-x} approx x for small x > 0.
    #
    # In addition to the numerically stable derivation above, we clamp
    # small/large values to be congruent with the logic in:
    # tensorflow/core/kernels/softplus_op.h
    #
    # Finally, we set the input to one whenever the input is too large or too
    # small. This ensures that no unchosen codepath is +/- inf. This is
    # necessary to ensure the gradient doesn't get NaNs. Recall that the
    # gradient of `where` behaves like `pred*pred_true + (1-pred)*pred_false`
    # thus an `inf` in an unselected path results in `0*inf=nan`. We are careful
    # to overwrite `x` with ones only when we will never actually use this
    # value. Note that we use ones and not zeros since `log(expm1(0.)) = -inf`.
    threshold = jnp.log(jnp.finfo(jnp.asarray(x).dtype).eps) + 2.

    is_too_small = x < jnp.exp(threshold)
    is_too_large = x > -threshold
    too_small_value = threshold
    too_large_value = x
    # This `where` will ultimately be a NOP because we won't select this
    # codepath whenever we used the surrogate `ones_like`.
    x = jnp.where(is_too_small | is_too_large, 1., x)
    y = x + jnp.log(-jnp.expm1(-x))  # == log(expm1(x))
    return jnp.where(is_too_small, too_small_value, jnp.where(is_too_large, too_large_value, y))


def _norm_except_axis(v: Array, norm: Callable[[Array], Scalar], axis: Optional[int]):
    if axis is None:
        return norm(v)
    else:
        return jax.vmap(norm, in_axes=axis, out_axes=axis)(v)


class WeightNorm(Module):
    layer: enn.Linear
    weight_name: str = eqx.field(static=True)
    axis: Optional[int] = eqx.field(static=True)

    def __init__(
        self,
        layer: enn.Linear,
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

    def _norm(self, w):
        return _norm_except_axis(w, norm=partial(jnp.linalg.norm, keepdims=True), axis=self.axis)
    
    @property
    def weight(self) -> Array:
        w = getattr(self.layer, self.weight_name)
        w = w / (self._norm(w) + _MIN_NORM)

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
        layer = eqx.tree_at(
            lambda layer: getattr(layer, self.weight_name), self.layer, weight
        )
        return layer(x)


class StationaryLinear(Module):
    layer: Module

    def __init__(self, state_dim, observation_dim, *, key, norm_readout: bool = False):
        self.layer = enn.Linear(state_dim, observation_dim, key=key, use_bias=True)

        if norm_readout:
            self.layer = WeightNorm(self.layer)

    def __call__(self, idx, x):
        return self.layer(x)

    @property
    def weight(self):
        return self.layer.weight


class VariantBiasLinear(Module):
    biases: Array
    layer: Module

    def __init__(
        self, state_dim, observation_dim, n_biases, *, key, norm_readout: bool = False
    ):
        wkey, bkey = jrandom.split(key, 2)

        self.layer = enn.Linear(state_dim, observation_dim, key=wkey, use_bias=False)
        lim = 1 / math.sqrt(state_dim)
        self.biases = jrandom.uniform(
            bkey,
            (n_biases, observation_dim),
            dtype=self.layer.weight.dtype,
            minval=-lim,
            maxval=lim,
        )

        if norm_readout:
            self.layer = WeightNorm(self.layer)

    def __call__(self, idx, x):
        x = self.layer(x)
        return x + self.biases[idx]

    @property
    def weight(self):
        return self.layer.weight

    def set_static(self, static=True) -> None:
        pass


def gauss_rbf(x, c, s):
    return jnp.exp(-jnp.sum(jnp.square((x - c)) * s))


class RBFN(Module):
    centers: Array = eqx.field(static=True)
    scale: Array
    readout: Module

    def __init__(
        self, input_size, output_size, network_size, *, key, normalized: bool = False
    ):
        key, ckey = jrandom.split(key)
        self.centers = jrandom.uniform(
            ckey, shape=(network_size, input_size), minval=-1, maxval=1
        )
        self.scale = jnp.ones(input_size)
        self.readout = enn.Linear(network_size, output_size, key=key)

    def __call__(self, x):
        kernels = jax.vmap(gauss_rbf, in_axes=(None, 0, None))(
            x, self.centers, constrain_positive(self.scale)
        )
        return self.readout(kernels)


class DataMask(Module, strict=True):
    p: float
    inference: bool

    def __init__(
        self,
        p: float = 0.5,
        inference: bool = False,
    ):

        self.p = p
        self.inference = inference

    @jax.named_scope("xfads.nn.DataMask")
    def __call__(
        self,
        x: Array,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Array:
        
        if inference is None:
            inference = self.inference

        if isinstance(self.p, (int, float)) and self.p == 0:
            inference = True
        
        shape = x.shape[:2] + (1,)  # broadcast to the last dimension
        if inference:
            return key, jnp.ones(shape)
        elif key is None:
            raise RuntimeError(
                f"{self.__name__} requires a key when running in non-deterministic mode."
            )
        else:
            key, subkey = jrandom.split(key)
            q  = 1 - jax.lax.stop_gradient(self.p)
            mask = jrandom.bernoulli(key, q, shape) 
            return subkey, mask


class Constrained(Module):
    array: Array
    transform: Callable
    inv_transform: Callable
    
    def __init__(self, value, transform: Callable, inv_transform: Callable):
        self.transform = transform
        self.inv_transform = inv_transform

        self.array = inv_transform(jnp.array(value))

    def __call__(self) -> Array:
        return self.transform(self.array)
