from typing import Protocol, Type

import jax
from jax import numpy as jnp, nn as jnn
from jaxtyping import Array, PRNGKeyArray, Scalar
import tensorflow_probability.substrates.jax.distributions as tfp
import chex
import equinox as eqx

from .nn import softplus_inverse, VariantBiasLinear, StationaryLinear
from .distribution import MVN
from .helper import Registry


MAX_LOGRATE = 15.
registry = Registry()


def get_class(name) -> Type:
    return registry.get_class(name)


class Observation(Protocol):
    def eloglik(self, key: PRNGKeyArray, t: Array, moment: Array, y: Array, approx, mc_size: int): ...

    def set_static(self, static=True): ...

    def initialize(self, *args, **kwargs): ...

    def init(self, *args, **kwargs): ...


@registry.register()
class Poisson(eqx.Module):
    readout: eqx.Module

    def __init__(self, state_dim, observation_dim, *, key, norm_readout: bool = False, **kwargs):
        n_steps = kwargs.get('n_steps', 0)

        if n_steps > 0:
            self.readout = VariantBiasLinear(state_dim, observation_dim, n_steps, key=key, norm_readout=norm_readout)
        else:
            self.readout = StationaryLinear(state_dim, observation_dim, key=key, norm_readout=norm_readout)
    
    def set_static(self, static=True) -> None:
        self.readout.set_static(static)

    def eloglik(self, key: PRNGKeyArray, t: Array, moment: Array, y: Array, approx, mc_size: int) -> Array:
        mean_z, cov_z = approx.moment_to_canon(moment)
        eta = self.readout(t, mean_z)
        V = jnp.diag(cov_z)
        C = self.readout.weight
        cvc = jnp.diag(C @ V @ C.T)
        loglam = jnp.minimum(eta + 0.5 * cvc, MAX_LOGRATE)
        lam = jnp.exp(loglam)
        ll = jnp.sum(y*eta - lam)
        ll = jnp.nan_to_num(ll)
        return ll


@registry.register()
class DiagGaussian(eqx.Module):
    unconstrained_cov: Array = eqx.field(static=False)
    readout: eqx.Module

    def __init__(self, state_dim, observation_dim, *, key, norm_readout: bool = False, **kwargs):
        cov = kwargs.get('cov', jnp.ones(observation_dim))
        self.unconstrained_cov = softplus_inverse(cov)

        n_steps = kwargs.get('n_steps', 0)

        if n_steps > 0:
            self.readout = VariantBiasLinear(state_dim, observation_dim, n_steps, key=key, norm_readout=norm_readout)
        else:
            self.readout = StationaryLinear(state_dim, observation_dim, key=key, norm_readout=norm_readout)

    def cov(self):
        return jnn.softplus(self.unconstrained_cov)

    def eloglik(self, key: PRNGKeyArray, t: Array, moment: Array, y: Array, approx: Type[MVN], mc_size: int) -> Array:
        mean_z, cov_z = approx.moment_to_canon(moment)
        mean_y = self.readout(t, mean_z)
        C = self.readout.weight  # left matrix
        cov_y = C @ approx.full_cov(cov_z) @ C.T + jnp.diag(self.cov())
        ll = tfp.MultivariateNormalFullCovariance(mean_y, cov_y).log_prob(y)
        return ll

    def set_static(self, static=True) -> None:
        self.__dataclass_fields__['unconstrained_cov'].metadata = {'static': static}

    def init(self, data):
        pass
