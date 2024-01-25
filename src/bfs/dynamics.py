from dataclasses import field

import jax
from jax import numpy as jnp, nn as jnn
from jaxtyping import Array, PRNGKeyArray
from equinox import Module

from .nn import make_mlp, softplus_inverse
from .distribution import ExponentialFamily

# TODO: decouple noise and transition
# model(likelihood, transition, noise, approximate, data)
# Decouple sample and deterministic
# Module as container of trainable arrays
# Never nest modules


class GaussianStateNoise(Module):
    unconstrained_cov: Array
    
    def __init__(self, cov):
        self.unconstrained_cov = softplus_inverse(cov)

    def cov(self) -> Array:
        return jnn.softplus(self.unconstrained_cov)


class Nonlinear(Module):
    forward: Module = field(init=False)

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        hidden_size: int,
        n_layers: int,
        *,
        key: PRNGKeyArray,
    ):
        self.forward = make_mlp(
            state_dim + input_dim, state_dim, hidden_size, n_layers, key=key
        )

    def __call__(self, z: Array, u: Array) -> Array:
        x = jnp.concatenate((z, u), axis=-1)
        return z + self.forward(x)


def predict_moment(
    z: Array, u: Array, forward, noise: GaussianStateNoise, approx: ExponentialFamily
) -> Array:
    """mu[t](z[t-1])"""
    ztp1 = forward(z, u)
    moment = approx.canon_to_moment(ztp1, noise.cov())
    return moment


def sample_expected_moment(
    key: PRNGKeyArray,
    moment: Array,
    u: Array,
    forward,
    noise: GaussianStateNoise,
    approx: ExponentialFamily,
    mc_size: int,
) -> Array:
    """E[mu[t](z[t-1])]"""
    z = approx.sample_by_moment(key, moment, mc_size)
    u_shape = (mc_size,) + u.shape
    u = jnp.broadcast_to(u, shape=u_shape)
    f = jax.vmap(
        lambda x, y: predict_moment(x, y, forward, noise, approx), in_axes=(0, 0)
    )
    moment = jnp.mean(f(z, u), axis=0)
    return moment
