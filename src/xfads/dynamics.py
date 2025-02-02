from dataclasses import field
from functools import partial

import jax
from jax import numpy as jnp, nn as jnn, random as jrandom
from jaxtyping import Array, PRNGKeyArray
import equinox as eqx
from equinox import Module, nn as enn

from .nn import make_mlp, softplus_inverse, RBFN
from .distribution import MVN

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


class Diffusion(GaussianStateNoise):
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


class Nonlinear(GaussianStateNoise):
    forward: Module = field(init=False)

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

    def __call__(self, z: Array, u: Array, reverse: bool = False) -> Array:
        x = jnp.concatenate((z, u), axis=-1)
        return self.forward(x)
    

class RBFNLinear(GaussianStateNoise):
    forward: Module = field(init=False)
    drive: Module = field(init=False)
    
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


    def __call__(self, z: Array, u: Array, reverse: bool = False) -> Array:
        return z + self.forward(z) + self.drive(u)


class LinearRecurrent(GaussianStateNoise):
    recurrent: Module = field(init=False)
    drive: Module = field(init=False)

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
            state_dim + input_dim, state_dim, width, depth, key=input_key
        )
        self.recurrent = enn.Linear(state_dim, state_dim, use_bias=False, key=recurrent_key)

    def __call__(self, z: Array, u: Array) -> Array:
        x = jnp.concatenate((z, u), axis=-1)
        return self.recurrent(z) + self.drive(x)
    

class LocallyLinearInput(GaussianStateNoise):
    B_shape: tuple[int, int] = eqx.field(static=True, init=False)
    recurrent: Module = eqx.field(init=False)
    drive: Module = eqx.field(init=False)

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
    

class LinearInput(GaussianStateNoise):
    recurrent: Module = field(init=False)
    drive: Module = field(init=False)

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
    

def build_dynamics(name: str, *, key, **kwargs) -> Module:
    dynamics_class = globals()[name]
    return dynamics_class(key=key, **kwargs)


def predict_moment(
    z: Array, u: Array, f, noise: GaussianStateNoise, approx: MVN, reverse: bool
) -> Array:
    """mu[t](z[t-1])"""
    ztp1 = f(z, u, reverse)
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
    noise: GaussianStateNoise,
    approx: MVN,
    mc_size: int,
    reverse: bool = False,
) -> Array:
    """E[mu[t](z[t-1])]"""
    z = approx.sample_by_moment(key, moment, mc_size)
    u_shape = (mc_size,) + u.shape
    u = jnp.broadcast_to(u, shape=u_shape)
    vf = jax.vmap(
        partial(predict_moment, f=f, noise=noise, approx=approx, reverse=reverse), in_axes=(0, 0)
    )
    moment = jnp.mean(vf(z, u), axis=0)
    return moment
