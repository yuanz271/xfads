from functools import partial
from typing import Protocol, Type

import jax
from jax import numpy as jnp, nn as jnn, random as jrandom
from jaxtyping import Array, PRNGKeyArray, Scalar
import equinox as eqx

from .helper import Registry
from .nn import softplus, softplus_inverse
from .distributions import MVN


registry = Registry()


def get_class(name) -> Type:
    return registry.get_class(name)


class Noise(Protocol):
    def cov(self): ...


def predict_moment(
    z: Array, u: Array, f, noise: Noise, approx: Type[MVN], *, key=None
) -> Array:
    """mu[t](z[t-1])"""
    ztp1 = f(z, u, key=key)
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
    key, subkey = jrandom.split(key)
    z = approx.sample_by_moment(subkey, moment, mc_size)
    u_shape = (mc_size,) + u.shape
    u = jnp.broadcast_to(u, shape=u_shape)
    f_vmap_sample_axis = jax.vmap(
        partial(predict_moment, f=f, noise=noise, approx=approx, key=key), in_axes=(0, 0)
    )
    moment = jnp.mean(f_vmap_sample_axis(z, u), axis=0)
    return moment


class DiagGaussian(eqx.Module, strict=True):
    unconstrained_cov: Array
    
    def __init__(self, cov, size):
        self.unconstrained_cov = jnp.full(size, fill_value=softplus_inverse(cov))

    def cov(self) -> Array:
        return softplus(self.unconstrained_cov)

    # def set_static(self, static=True) -> None:
    #     self.__dataclass_fields__['unconstrained_cov'].metadata = {'static': static}


class AbstractDynamics(eqx.Module):
    def cov(self) -> Array:
        return self.noise.cov()
    
    def loss(self) -> Scalar:
        return 0.


# @registry.register()
# class Diffusion(AbstractDynamics):
#     unconstrained_cov: Array

#     def __init__(
#         self,
#         state_dim: int,
#         input_dim: int,
#         width: int,
#         depth: int,
#         cov,
#         *,
#         key: PRNGKeyArray,
#         **kwargs,
#     ):
#         self.unconstrained_cov = jnp.full(state_dim, fill_value=jax.scipy.special.logit(cov))

#     def __call__(self, z: Array, u: Array) -> Array:
#         return jnp.sqrt(1 - self.cov()) * z

#     def cov(self) -> Array:
#         return jnn.sigmoid(self.unconstrained_cov)
