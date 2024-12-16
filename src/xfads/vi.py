from typing import Callable, ClassVar, Type

import jax
from jax import numpy as jnp, nn as jnn
from jaxtyping import Array, PRNGKeyArray, Scalar
from equinox import Module
import tensorflow_probability.substrates.jax.distributions as tfp
import chex
import equinox as eqx

from .nn import WeightNorm, softplus_inverse, VariantBiasLinear
from .distribution import MVN


class Likelihood(Module):
    readout: ClassVar[Callable]
    eloglik: ClassVar[Callable]
        

class PoissonLik(Likelihood):
    readout: Module

    def __init__(self, readout, norm_readout: bool = False):
        self.readout = readout
        if norm_readout:
            self.readout = WeightNorm(self.readout)
    
    def eloglik(self, key: PRNGKeyArray, t: Array, moment: Array, y: Array, mc_size: int) -> Array:
        y = jnp.broadcast_to(y, (mc_size,) + y.shape)
        chex.assert_shape(y, (mc_size, None))
        z = MVN.sample_by_moment(key, moment, mc_size)
        eta = jax.vmap(self.readout)(z)
        rate = jnn.softplus(eta)
        ll = tfp.Poisson(rate=rate).log_prob(y)
        return jnp.mean(ll, axis=0)


class NonstationaryPoissonLik(Likelihood):
    readout: Module

    def __init__(self, state_dim, observation_dim, n_steps, *, key, norm_readout: bool = False) -> None:
        self.readout = VariantBiasLinear(state_dim, observation_dim, n_steps, key=key, norm_readout=norm_readout)
    
    def predict(self, key: PRNGKeyArray, t: Array, moment: Array, mc_size: int) -> Array:
        t = jnp.broadcast_to(t, (mc_size,) + t.shape)
        z = MVN.sample_by_moment(key, moment, mc_size)
        eta = jax.vmap(self.readout)(t, z)  # Monte Carlo dimension
        rate = jnn.softplus(eta)
        return rate

    def eloglik(self, key: PRNGKeyArray, t: Array, moment: Array, y: Array, mc_size: int) -> Array:
        y = jnp.broadcast_to(y, (mc_size,) + y.shape)
        chex.assert_shape(y, (mc_size, None))
        # z = MVN.sample_by_moment(key, moment, mc_size)
        # eta = jax.vmap(self.readout)(z)
        # rate = jnn.softplus(eta)
        rate = self.predict(key, t, moment, mc_size)
        ll = tfp.Poisson(rate=rate).log_prob(y)
        return jnp.mean(ll, axis=0)


# class GaussainLik(Likelihood):
#     unconstrained_cov: Array
#     readout: Module

#     def __init__(self, cov, readout):
#         self.unconstrained_cov = jnp.linalg.cholesky(cov)
#         self.readout = readout

#     def cov(self):
#         return self.unconstrained_cov @ self.unconstrained_cov.T 
    
#     def eloglik(self, key: PRNGKeyArray, moment: Array, y: Array, mc_size: int) -> Array:
#         mean_z, cov_z = MVN.moment_to_canon(moment)
#         mean_y = self.readout(mean_z)
#         loading = self.readout.weight  # left matrix
#         cov_y = loading @ cov_z @ loading.T + self.cov()
#         ll = tfp.MultivariateNormalFullCovariance(mean_y, cov_y).log_prob(y)
#         return ll

#     def set_static(self, static=True) -> None:
#         self.__dataclass_fields__['unconstrained_cov'].metadata = {'static': static}


class DiagMVNLik(Likelihood):
    unconstrained_cov: Array = eqx.field(static=False)
    # _cov: Array = eqx.field(static=True)
    readout: Module
    # dim: int

    def __init__(self, cov, readout, norm_readout: bool = False):
        self.unconstrained_cov = softplus_inverse(cov)
        self.readout = readout
        if norm_readout:
            self.readout = WeightNorm(self.readout)

    def cov(self):
        return jnn.softplus(self.unconstrained_cov)

    def eloglik(self, key: PRNGKeyArray, t: Array, moment: Array, y: Array, approx: Type[MVN], mc_size: int) -> Array:
        mean_z, cov_z = approx.moment_to_canon(moment)
        mean_y = self.readout(mean_z)
        loading = self.readout.weight  # left matrix
        cov_y = loading @ approx.full_cov(cov_z) @ loading.T + jnp.diag(self.cov())
        ll = tfp.MultivariateNormalFullCovariance(mean_y, cov_y).log_prob(y)
        return ll

    def set_static(self, static=True) -> None:
        self.__dataclass_fields__['unconstrained_cov'].metadata = {'static': static}


def elbo(key: PRNGKeyArray, t: Array, moment: Array, moment_p: Array, y: Array, eloglik: Callable[..., Scalar], approx: Type[MVN], *, mc_size: int) -> Scalar:
    """Single time point"""
    ell: Scalar = eloglik(key, t, moment, y, approx, mc_size)
    kl: Scalar = approx.kl(moment, moment_p)
    
    lval: Scalar = ell - kl
    
    return lval
