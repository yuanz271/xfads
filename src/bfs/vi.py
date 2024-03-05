from typing import Callable, ClassVar, Type
import jax
from jax import numpy as jnp, nn as jnn
from jaxtyping import Array, PRNGKeyArray, Scalar
from equinox import Module
import tensorflow_probability.substrates.jax.distributions as tfp
import chex
import equinox as eqx

from .nn import softplus_inverse
from .distribution import MVN, DiagMVN, ExponentialFamily


class Likelihood(Module):
    readout: ClassVar[Callable]
    eloglik: ClassVar[Callable]


class PoissonLik(Likelihood):
    readout: Module

    def eloglik(self, key: PRNGKeyArray, moment: Array, y: Array, mc_size: int) -> Array:
        y = jnp.broadcast_to(y, (mc_size,) + y.shape)
        chex.assert_shape(y, (mc_size, None))
        z = MVN.sample_by_moment(key, moment, mc_size)
        eta = jax.vmap(self.readout)(z)
        rate = jnn.softplus(eta)
        ll = tfp.Poisson(rate=rate).log_prob(y)
        return jnp.mean(ll, axis=0)


class GaussainLik(Likelihood):
    unconstrained_cov: Array
    readout: Module

    def __init__(self, cov, readout):
        self.unconstrained_cov = jnp.linalg.cholesky(cov)
        self.readout = readout

    def cov(self):
        return self.unconstrained_cov @ self.unconstrained_cov.T 
    
    def eloglik(self, key: PRNGKeyArray, moment: Array, y: Array, mc_size: int) -> Array:
        mean_z, cov_z = MVN.moment_to_canon(moment)
        mean_y = self.readout(mean_z)
        loading = self.readout.weight  # left matrix
        cov_y = loading @ cov_z @ loading.T + self.cov()
        ll = tfp.MultivariateNormalFullCovariance(mean_y, cov_y).log_prob(y)
        return ll


class DiagGaussainLik(Likelihood):
    unconstrained_cov: Array = eqx.field(static=True)
    # _cov: Array = eqx.field(static=True)
    readout: Module
    # dim: int

    def __init__(self, cov, readout):
        self.unconstrained_cov = softplus_inverse(cov)
        self.readout = readout
        # self.dim = jnp.size(readout.bias)
        # self._cov = cov

    def cov(self):
        return jnn.softplus(self.unconstrained_cov)
        # return jnp.ones(self.dim) * self.s
        # return self._cov

    def eloglik(self, key: PRNGKeyArray, moment: Array, y: Array, mc_size: int) -> Array:
        mean_z, cov_z = DiagMVN.moment_to_canon(moment)
        mean_y = self.readout(mean_z)
        loading = self.readout.weight  # left matrix
        cov_y = loading @ jnp.diag(cov_z) @ loading.T + jnp.diag(self.cov())
        ll = tfp.MultivariateNormalFullCovariance(mean_y, cov_y).log_prob(y)
        return ll


def elbo(key: PRNGKeyArray, moment: Array, moment_p: Array, y: Array, eloglik: Callable[..., Scalar], approx: Type[ExponentialFamily], *, mc_size: int) -> Scalar:
    """Single time point"""
    ell: Scalar = eloglik(key, moment, y, mc_size)
    kl: Scalar = approx.kl(moment, moment_p)
    
    lval: Scalar = ell - kl
    
    return lval
