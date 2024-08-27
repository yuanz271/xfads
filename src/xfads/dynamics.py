from dataclasses import field

import jax
from jax import numpy as jnp, nn as jnn, random as jrandom
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx
from equinox import Module, nn as enn

from .nn import make_mlp, softplus_inverse
from .distribution import ExponentialFamily, DiagMVN, FullMVN, LoRaMVN

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


class Diffusion(Module):
    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        width: int,
        depth: int,
        *,
        key: PRNGKeyArray,
    ):
        pass

    def __call__(self, z: Array, u: Array) -> Array:
        return z


class Nonlinear(Module):
    forward: Module = field(init=False)

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        width: int,
        depth: int,
        *,
        key: PRNGKeyArray,
    ):
        self.forward = make_mlp(
            state_dim + input_dim, state_dim, width, depth, key=key
        )

    def __call__(self, z: Array, u: Array) -> Array:
        x = jnp.concatenate((z, u), axis=-1)
        return self.forward(x)


class LinearRecurrent(Module):
    recurrent: Module = field(init=False)
    drive: Module = field(init=False)

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        width: int,
        depth: int,
        *,
        key: PRNGKeyArray,
    ):
        recurrent_key, input_key = jrandom.split(key)
        self.drive = make_mlp(
            state_dim + input_dim, state_dim, width, depth, key=input_key
        )
        self.recurrent = enn.Linear(state_dim, state_dim, use_bias=False, key=recurrent_key)

    def __call__(self, z: Array, u: Array) -> Array:
        x = jnp.concatenate((z, u), axis=-1)
        return self.recurrent(z) + self.drive(x)
    

class LinearInput(Module):
    recurrent: Module = field(init=False)
    drive: Module = field(init=False)

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        width: int,
        depth: int,
        *,
        key: PRNGKeyArray,
    ):
        recurrent_key, input_key = jrandom.split(key)
        self.recurrent = make_mlp(
            state_dim, state_dim, width, depth, key=recurrent_key
        )
        self.drive = enn.Linear(input_dim, state_dim, use_bias=False, key=input_key)

    def __call__(self, z: Array, u: Array) -> Array:
        return self.recurrent(z) + self.drive(u)
    

def build_dynamics(name: str, *, key, **kwargs) -> Module:
    dynamics_class = globals()[name]
    return dynamics_class(key=key, **kwargs)


def predict_moment(
    z: Array, u: Array, forward, noise: GaussianStateNoise, approx: ExponentialFamily
) -> Array:
    """mu[t](z[t-1])"""
    ztp1 = forward(z, u)
    match approx():
        case DiagMVN():
            M2 = noise.cov()
        case FullMVN():
            M2 = jnp.diag(noise.cov())
        case LoRaMVN():
            M2 = jnp.diag(noise.cov())
        case _:
            raise ValueError(f"{approx}")
    moment = approx.canon_to_moment(ztp1, M2)
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
