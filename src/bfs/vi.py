from abc import ABCMeta, abstractmethod
from typing import Any, Callable, ClassVar
import jax
from jax import numpy as jnp, nn as jnn, random as jrnd
from jaxtyping import Array, PRNGKeyArray, Scalar
from equinox import Module, nn as enn
import tensorflow_probability.substrates.jax.distributions as tfp
import chex

from .distribution import MVN, DiagMVN


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
    cov: Array
    readout: Module

    def eloglik(self, key: PRNGKeyArray, moment: Array, y: Array, mc_size: int) -> Array:
        mean_z, cov_z = MVN.moment_to_canon(moment)
        mean_y = self.readout(mean_z)
        loading = self.readout.weight  # left matrix
        cov_y = loading @ cov_z @ loading.T + self.cov
        ll = tfp.MultivariateNormalFullCovariance(mean_y, cov_y).log_prob(y)
        return ll


class DiagGaussainLik(Likelihood):
    cov: Array
    readout: Module

    def eloglik(self, key: PRNGKeyArray, moment: Array, y: Array, mc_size: int) -> Array:
        mean_z, cov_z = DiagMVN.moment_to_canon(moment)
        mean_y = self.readout(mean_z)
        loading = self.readout.weight  # left matrix
        cov_y = loading @ jnp.diag(cov_z) @ loading.T + jnp.diag(self.cov)
        ll = tfp.MultivariateNormalFullCovariance(mean_y, cov_y).log_prob(y)
        return ll


def elbo(key, moment, moment_p, y, eloglik, approx, *, mc_size, negative=True) -> Scalar:
    """Single time point"""
    ell = eloglik(key, moment, y, mc_size)
    kl = approx.kl(moment, moment_p)
    
    lval = ell - kl
    
    if negative:
        lval = -lval
    
    return lval
