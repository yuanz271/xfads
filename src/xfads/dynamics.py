from functools import partial
from typing import Type

import jax
from jax import numpy as jnp, nn as jnn, random as jrandom
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx
from equinox import Module, nn as enn

from .helper import SingletonMeta
from .nn import make_mlp, softplus_inverse, RBFN
from .distribution import MVN

# TODO: decouple noise and transition
# model(likelihood, transition, noise, approximate, data)
# Decouple sample and deterministic
# Module as container of trainable arrays
# Never nest modules

class Registry(metaclass=SingletonMeta):
    _registry: dict = {}  # class variable

    @classmethod
    def register(cls, name=None):

        def decorate(klass):
            nonlocal name
            if name is None:
                name = klass.__name__
            cls._registry[name] = klass
            return klass
        
        return decorate

    @classmethod
    def get_class(cls, name):
        klass = cls._registry[name]
        return klass


class DiagNoise(Module):
    unconstrained_cov: Array = eqx.field(static=False)
    
    def __init__(self, cov):
        self.unconstrained_cov = softplus_inverse(cov)

    def cov(self) -> Array:
        return jnn.softplus(self.unconstrained_cov)

    def set_static(self, static=True) -> None:
        self.__dataclass_fields__['unconstrained_cov'].metadata = {'static': static}


@Registry.register()
class Diffusion(DiagNoise):
    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        width: int,
        depth: int,
        cov,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(cov)

    def __call__(self, z: Array, u: Array) -> Array:
        return z


@Registry.register()
class Nonlinear(DiagNoise):
    forward: Module

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        width: int,
        depth: int,
        cov,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(cov)
        self.forward = make_mlp(
            state_dim + input_dim, state_dim, width, depth, key=key
        )

    def __call__(self, z: Array, u: Array) -> Array:
        x = jnp.concatenate((z, u), axis=-1)
        return self.forward(x)
    

@Registry.register()
class RBFNLinear(DiagNoise):
    forward: Module 
    drive: Module
    
    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        width: int,
        depth: int,
        cov,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(cov)

        fkey, dkey = jrandom.split(key)
        self.forward = RBFN(
            state_dim, state_dim, width, key=fkey
        )
        self.drive = enn.Linear(input_dim, state_dim, use_bias=False, key=dkey)


    def __call__(self, z: Array, u: Array) -> Array:
        return z + self.forward(z) + self.drive(u)


@Registry.register()
class LocallyLinearInput(DiagNoise):
    B_shape: tuple[int, int] = eqx.field(static=True)
    recurrent: Module 
    drive: Module 

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        width: int,
        depth: int,
        cov,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(cov)
        recurrent_key, input_key = jrandom.split(key)
        self.drive = make_mlp(
            state_dim, state_dim * input_dim, width, depth, key=input_key
        )
        self.recurrent = enn.Linear(state_dim, state_dim, use_bias=False, key=recurrent_key)
        self.B_shape = (state_dim, input_dim)

    def __call__(self, z: Array, u: Array) -> Array:
        vec_B = self.drive(z)
        B = jnp.reshape(vec_B, self.B_shape)
        return self.recurrent(z) + B @ u
    

@Registry.register()
class LinearInput(DiagNoise):
    recurrent: Module 
    drive: Module 

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        width: int,
        depth: int,
        cov,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(cov)
        recurrent_key, input_key = jrandom.split(key)
        self.recurrent = make_mlp(
            state_dim, state_dim, width, depth, key=recurrent_key
        )
        self.drive = enn.Linear(input_dim, state_dim, use_bias=False, key=input_key)

    def __call__(self, z: Array, u: Array) -> Array:
        return self.recurrent(z) + self.drive(u)


def predict_moment(
    z: Array, u: Array, f, noise: DiagNoise, approx: Type[MVN]
) -> Array:
    """mu[t](z[t-1])"""
    ztp1 = f(z, u)
    M2 = approx.noise_moment(noise.cov())
    # match approx():
    #     case DiagMVN():
    #         M2 = noise.cov()
    #     case FullMVN():
    #         M2 = jnp.diag(noise.cov())
    #     case LoRaMVN():
    #         M2 = jnp.diag(noise.cov())
    #     case _:
    #         raise ValueError(f"{approx}")
    moment = approx.canon_to_moment(ztp1, M2)
    return moment


def sample_expected_moment(
    key: PRNGKeyArray,
    moment: Array,
    u: Array,
    f,
    noise: DiagNoise,
    approx: Type[MVN],
    mc_size: int,
) -> Array:
    """E[mu[t](z[t-1])]"""
    z = approx.sample_by_moment(key, moment, mc_size)
    u_shape = (mc_size,) + u.shape
    u = jnp.broadcast_to(u, shape=u_shape)
    vf = jax.vmap(
        partial(predict_moment, f=f, noise=noise, approx=approx), in_axes=(0, 0)
    )
    moment = jnp.mean(vf(z, u), axis=0)
    return moment
