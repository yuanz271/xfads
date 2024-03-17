import jax
from jax import numpy as jnp, nn as jnn, random as jrandom
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx
from equinox import Module, nn as enn

from .nn import make_mlp, softplus_inverse
from .distribution import ExponentialFamily

# TODO: decouple noise and transition
# model(likelihood, transition, noise, approximate, data)
# Decouple sample and deterministic
# Module as container of trainable arrays
# Never nest modules


class GaussianStateNoise(Module):
    unconstrained_cov: Array = eqx.field(static=False)
    
    def __init__(self, cov):
        self.unconstrained_cov = softplus_inverse(cov)

    def cov(self) -> Array:
        return jnn.softplus(self.unconstrained_cov)
    
    def set_static(self, static=True) -> None:
        self.__dataclass_fields__['unconstrained_cov'].metadata = {'static': static}


class Nonlinear(Module):
    forward: Module = eqx.field(init=False)

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        *,
        key: PRNGKeyArray,
        kwargs: dict
    ):
        hidden_size: int = kwargs['width']
        n_layers: int = kwargs['depth']

        self.forward = make_mlp(
            state_dim + input_dim, state_dim, hidden_size, n_layers, key=key
        )

    def __call__(self, z: Array, u: Array) -> Array:
        x = jnp.concatenate((z, u), axis=-1)
        return z + self.forward(x)


class Linear(Module):
    autonomous: Module = eqx.field(init=False)
    control: Module = eqx.field(init=False)

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        *,
        key: PRNGKeyArray,
        kwargs: dict
    ):
        akey, ikey = jrandom.split(key)
        self.autonomous = enn.Linear(state_dim, state_dim, key=akey)
        self.control = enn.Linear(input_dim, state_dim, use_bias=False, key=akey)

    def __call__(self, z: Array, u: Array) -> Array:
        return self.autonomous(z) + self.control(u)


class Diffusion(Module):
    decay: float = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        *,
        key: PRNGKeyArray,
        kwargs: dict
    ):
        self.decay = 1.

    def __call__(self, z: Array, u: Array) -> Array:
        return z


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
