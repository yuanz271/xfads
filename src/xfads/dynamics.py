from functools import partial
from typing import Type

import jax
from jax import numpy as jnp, nn as jnn, random as jrandom
from jaxtyping import Array, PRNGKeyArray, Scalar
import equinox as eqx
from equinox import Module, nn as enn

from .helper import Registry
from .nn import make_mlp, softplus, softplus_inverse, RBFN
from .distribution import MVN


registry = Registry()


def get_class(name) -> Type:
    return registry.get_class(name)
    

class Noise(Module):
    unconstrained_cov: Array
    
    def __init__(self, cov, size):
        self.unconstrained_cov = softplus_inverse(cov * jnp.ones(size))

    def cov(self) -> Array:
        return softplus(self.unconstrained_cov)

    # def set_static(self, static=True) -> None:
    #     self.__dataclass_fields__['unconstrained_cov'].metadata = {'static': static}


class Isotropic(Module):
    unconstrained_var: Array
    size: int = eqx.field(static=True)
    
    def __init__(self, var, size):
        self.unconstrained_var = jax.scipy.special.logit(var)
        self.size = size

    def var(self) -> Scalar:
        return jnn.sigmoid(self.unconstrained_var)

    def cov(self) -> Array:
        return self.var() * jnp.ones(self.size)


class AbstractDynamics(Module):
    noise: eqx.AbstractVar[Module]

    def cov(self) -> Array:
        return self.noise.cov()
    
    def loss(self) -> Scalar:
        0.


@registry.register()
class Diffusion(AbstractDynamics):
    noise: Module
    B: Module

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        width: int,
        depth: int,
        cov,
        *,
        key: PRNGKeyArray,
        **kwargs,
    ):
        self.noise = Isotropic(cov, state_dim)
        self.B = eqx.nn.Linear(input_dim, state_dim, use_bias=False, key=key)

    def __call__(self, z: Array, u: Array) -> Array:
        return jnp.sqrt(1 - self.noise.var()) * z + self.B(u)


@registry.register()
class Nonlinear(AbstractDynamics):
    noise: Module
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
        **kwargs,
    ):
        self.noise = Noise(cov, state_dim)

        self.forward = make_mlp(
            state_dim + input_dim, state_dim, width, depth, key=key
        )

    def __call__(self, z: Array, u: Array) -> Array:
        x = jnp.concatenate((z, u), axis=-1)
        return self.forward(x)
    

@registry.register()
class RBFNLinear(AbstractDynamics):
    noise: Module
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
        **kwargs,
    ):
        self.noise = Noise(cov, state_dim)

        fkey, dkey = jrandom.split(key)
        self.forward = RBFN(
            state_dim, state_dim, width, key=fkey
        )
        self.drive = enn.Linear(input_dim, state_dim, use_bias=False, key=dkey)


    def __call__(self, z: Array, u: Array) -> Array:
        return z + self.forward(z) + self.drive(u)


@registry.register()
class LocallyLinearInput(AbstractDynamics):
    noise: Module
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
        self.noise = Noise(cov, state_dim)
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
    

@registry.register()
class LinearInput(AbstractDynamics):
    noise: Module
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
        self.noise = Noise(cov, state_dim)
        recurrent_key, input_key = jrandom.split(key)
        self.recurrent = make_mlp(
            state_dim, state_dim, width, depth, key=recurrent_key
        )
        self.drive = enn.Linear(input_dim, state_dim, use_bias=False, key=input_key)

    def __call__(self, z: Array, u: Array) -> Array:
        return self.recurrent(z) + self.drive(u)


def predict_moment(
    z: Array, u: Array, f, noise: Noise, approx: Type[MVN]
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
    noise: Noise,
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
