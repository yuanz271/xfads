from collections.abc import Callable
from functools import partial
from typing import ClassVar, Optional, Type
import jax
from jax import numpy as jnp, nn as jnn, random as jrandom
from jaxtyping import Array, PRNGKeyArray, Scalar
from equinox import Module
import tensorflow_probability.substrates.jax.distributions as tfp
import chex
import equinox as eqx

from .nn import WeightNorm, softplus_inverse
from .distribution import ExponentialFamily


_poisson_link = jnn.softplus
_constrain_diag_cov = softplus_inverse


class Likelihood(Module):
    readout: ClassVar[Module]
    eloglik: ClassVar[Callable]


class PoissonLik(Likelihood):
    readout: Module

    def eloglik(
        self,
        key: PRNGKeyArray,
        moment: Array,
        covariate_predict: Array,
        y: Array,
        approx: Type[ExponentialFamily],
        mc_size: int,
    ) -> Array:
        y = jnp.broadcast_to(y, (mc_size,) + y.shape)
        covariate_predict = jnp.broadcast_to(
            covariate_predict, (mc_size,) + covariate_predict.shape
        )
        chex.assert_shape(y, (mc_size, None))
        z = approx.sample_by_moment(key, moment, mc_size)
        eta = jax.vmap(self.readout)(z) + covariate_predict
        rate = _poisson_link(eta)
        ll = tfp.Poisson(rate=rate).log_prob(y)
        return jnp.mean(ll, axis=0)


class MVNLik(Likelihood):
    unconstrained_cov: Array
    readout: WeightNorm

    def __init__(self, cov, readout) -> None:
        self.unconstrained_cov = jnp.linalg.cholesky(cov)
        self.readout = WeightNorm(readout)

    def cov(self) -> Array:
        return self.unconstrained_cov @ self.unconstrained_cov.T

    def eloglik(
        self,
        key: PRNGKeyArray,
        moment: Array,
        covariate_predict: Array,
        y: Array,
        approx: Type[ExponentialFamily],
        mc_size: int,
    ) -> Array:
        mean_z, cov_z = approx.moment_to_canon(moment)
        mean_y = self.readout(mean_z) + covariate_predict
        loading = self.readout.weight()  # left matrix
        cov_y = loading @ approx.full_cov(cov_z) @ loading.T + self.cov()
        ll = tfp.MultivariateNormalFullCovariance(mean_y, cov_y).log_prob(y)
        return ll


class DiagMVNLik(Likelihood):
    unconstrained_cov: Array = eqx.field(static=False)
    readout: WeightNorm

    def __init__(self, cov, readout) -> None:
        self.unconstrained_cov = softplus_inverse(cov)
        self.readout = WeightNorm(readout)

    def cov(self) -> Array:
        return _constrain_diag_cov(self.unconstrained_cov)

    def eloglik(
        self,
        key: PRNGKeyArray,
        moment: Array,
        covariate_predict: Array,
        y: Array,
        approx: Type[ExponentialFamily],
        mc_size: int,
    ) -> Array:
        mean_z, cov_z = approx.moment_to_canon(moment)
        mean_y = self.readout(mean_z) + covariate_predict
        loading = self.readout.weight()  # left matrix
        cov_y = loading @ approx.full_cov(cov_z) @ loading.T + jnp.diag(self.cov())
        ll = tfp.MultivariateNormalFullCovariance(mean_y, cov_y).log_prob(y)
        return ll

    def set_static(self, static=True) -> None:
        self.__dataclass_fields__["unconstrained_cov"].metadata = {"static": static}


def elbo(
    key: PRNGKeyArray,
    moment: Array,
    moment_p: Array,
    covariate_predict: Array,
    y: Array,
    eloglik: Callable[..., Scalar],
    approx: Type[ExponentialFamily],
    *,
    mc_size: int,
) -> Scalar:
    """Single time point"""
    ell: Scalar = eloglik(key, moment, covariate_predict, y, approx, mc_size)
    kl: Scalar = approx.kl(moment, moment_p)

    lval: Scalar = ell - kl

    return lval


def make_batch_elbo(
    eloglik, approx, mc_size
) -> Callable[[PRNGKeyArray, Array, Array, Array, Array, Optional[Callable]], Scalar]:
    batch_elbo = jax.vmap(
        jax.vmap(partial(elbo, eloglik=eloglik, approx=approx, mc_size=mc_size))
    )  # (batch, seq)

    def wrapper(
        key: PRNGKeyArray,
        moment_s: Array,
        moment_p: Array,
        covariate_predict: Array,
        ys: Array,
        *,
        reduce: Callable = jnp.mean,
    ) -> Scalar:
        keys = jrandom.split(key, ys.shape[:2])  # ys.shape[:2] + (2,)
        return reduce(batch_elbo(keys, moment_s, moment_p, covariate_predict, ys))

    return wrapper
