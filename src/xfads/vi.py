from typing import Callable, Protocol, Type

import jax
from jax import numpy as jnp, nn as jnn
from jaxtyping import Array, PRNGKeyArray, Scalar
from equinox import Module
import tensorflow_probability.substrates.jax.distributions as tfp
import chex
import equinox as eqx

from .nn import WeightNorm, softplus_inverse, VariantBiasLinear, StationaryLinear
from .distribution import MVN


MAX_LOGRATE = 15.


class Likelihood(Protocol):
    def eloglik(self, *args, **kwargs):
        pass
        

class PoissonLik(Module):
    readout: Module

    def __init__(self, state_dim, observation_dim, *, key, norm_readout: bool = False, **kwargs):
        n_steps = kwargs.get('n_steps', 0)
        biases = kwargs.get('biases', "none")

        if n_steps > 0:
            self.readout = VariantBiasLinear(state_dim, observation_dim, n_steps, biases, key=key, norm_readout=norm_readout)
        else:
            self.readout = StationaryLinear(state_dim, observation_dim, key=key, norm_readout=norm_readout)
    
    def set_static(self, static=True) -> None:
        if isinstance(self.readout, VariantBiasLinear):
            self.readout.set_static(static)

    # def eloglik(self, key: PRNGKeyArray, t: Array, moment: Array, y: Array, approx, mc_size: int) -> Array:
    #     # y = jnp.broadcast_to(y, (mc_size,) + y.shape)
    #     # chex.assert_shape(y, (mc_size, None))
    #     # z = DiagMVN.sample_by_moment(key, moment, mc_size)
    #     # eta = jax.vmap(self.readout)(z)
    #     # eta = jnp.clip(eta, max=MAX_ETA)
    #     # rate = jnp.exp(eta)
    #     # ll = tfp.Poisson(rate=rate).log_prob(y)
    #     # return jnp.mean(ll, axis=0)
    #     mean_z, cov_z = approx.moment_to_canon(moment)
    #     eta = self.readout(mean_z)
    #     eta = jnp.minimum(eta, MAX_ETA)
    #     lam = jnp.exp(eta)
    #     ll = jnp.sum(y*eta - lam)
    #     # ll = tfp.Poisson(log_rate=eta).log_prob(y)
    #     return ll

    def eloglik(self, key: PRNGKeyArray, t: Array, moment: Array, y: Array, approx, mc_size: int) -> Array:
        mean_z, cov_z = approx.moment_to_canon(moment)
        eta = self.readout(t, mean_z)
        V = jnp.diag(cov_z)
        C = self.readout.weight
        cvc = jnp.diag(C @ V @ C.T)
        loglam = jnp.minimum(eta + 0.5 * cvc, MAX_LOGRATE)
        lam = jnp.exp(loglam)
        ll = jnp.sum(y*eta - lam)
        # ll = tfp.Poisson(log_rate=eta).log_prob(y)
        return ll

# class NonstationaryPoissonLik(Likelihood):
#     readout: VariantBiasLinear

#     def __init__(self, state_dim, observation_dim, *, key, norm_readout: bool = False, **kwargs) -> None:
#         self.state_dim = state_dim
#         n_steps = kwargs['n_steps']
#         biases = kwargs.get('biases', "none") 
#         self.readout = VariantBiasLinear(state_dim, observation_dim, n_steps, biases, key=key, norm_readout=norm_readout)

#     def eloglik(self, key: PRNGKeyArray, t: Array, moment: Array, y: Array, approx, mc_size: int) -> Array:
#         mean_z, cov_z = approx.moment_to_canon(moment)
#         eta = self.readout(t, mean_z)
#         eta = jnp.minimum(eta, MAX_ETA)
#         lam = jnp.exp(eta)
#         ll = jnp.sum(y*eta - lam)
#         # ll = tfp.Poisson(log_rate=eta).log_prob(y)
#         return ll


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


class DiagMVNLik(Module):
    unconstrained_cov: Array = eqx.field(static=False)
    readout: Module

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
