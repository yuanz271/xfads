"""
Exponential-family variational distributions
"""

import math
from typing import Protocol, Type

import jax
from jax import numpy as jnp, random as jrandom
# from jax.nn import softplus  # this version is overflow safe
from jaxtyping import Array, Scalar, PRNGKeyArray
import tensorflow_probability.substrates.jax.distributions as tfp

from .helper import Registry
from .nn import constrain_positive, unconstrain_positive


MIN_VAR = 1e-6
EPS = 1e-6
registry = Registry()


def get_class(name) -> Type:
    return registry.get_class(name)


def _inv(a):
    return jnp.linalg.inv(a + EPS * jnp.eye(a.shape[-1]))


class Approx(Protocol):
    """Interface of MVN distributions
    The classes should implement class functions performing
    converion between natural parameter and mean parameter.
    """

    @classmethod
    def natural_to_moment(cls, natural) -> Array: ...

    @classmethod
    def moment_to_natural(cls, moment) -> Array: ...

    @classmethod
    def sample_by_moment(cls, key, moment, mc_size) -> Array: ...

    @classmethod
    def param_size(cls, state_dim) -> int: ...

    @classmethod
    def kl(cls, moment1, moment2) -> Scalar: ...

    @classmethod
    def moment_to_canon(cls, moment: Array) -> tuple[Array, Array]: ...

    @classmethod
    def canon_to_moment(cls, mean: Array, cov: Array) -> Array: ...

    @classmethod
    def full_cov(cls, cov: Array) -> Array: ...

    @classmethod
    def constrain_moment(cls, unconstrained: Array) -> Array: ...

    @classmethod
    def constrain_natural(cls, unconstrained: Array) -> Array: ...

    @classmethod
    def unconstrain_natural(cls, natural: Array) -> Array: ...

    @classmethod
    def noise_moment(cls, noise_cov) -> Array: ...


@registry.register()
class FullMVN:
    @classmethod
    def natural_to_moment(cls, natural: Array) -> Array:
        """Pmu, -P/2 => mu, P"""
        n = jnp.size(natural)
        m = cls.variable_size(n)
        nat1, nat2 = jnp.split(natural, [m])
        p = -2 * nat2  # vectorized precision
        P = jnp.reshape(p, (m, m))  # precision matrix
        loc = jnp.linalg.solve(P, nat1)
        V = _inv(P)
        v = V.flatten()
        moment = jnp.concatenate((loc, v))
        return moment

    @classmethod
    def moment_to_natural(cls, moment: Array) -> Array:
        loc, V = cls.moment_to_canon(moment)
        P = _inv(V)
        Nat2 = -0.5 * P
        nat2 = Nat2.flatten()
        nat1 = P @ loc
        natural = jnp.concatenate((nat1, nat2))
        return natural

    @classmethod
    def sample_by_moment(cls, key: PRNGKeyArray, moment: Array, mc_size: int) -> Array:
        loc, V = cls.moment_to_canon(moment)
        return jrandom.multivariate_normal(
            key, loc, V, shape=(mc_size,)
        )  # It seems JAX does reparameterization trick

    @classmethod
    def moment_to_canon(cls, moment: Array) -> tuple[Array, Array]:
        n = jnp.size(moment)
        m = cls.variable_size(n)
        loc, v = jnp.split(moment, [m])
        V = jnp.reshape(v, (m, m))
        return loc, V

    @classmethod
    def variable_size(cls, param_size: int) -> int:
        """
        Get the variable size given a parameter vector size
        """
        # n: size of vectorized mean param
        # m: size of random variable

        # n = m + m*m
        # m = (sqrt(1 + 4n) - 1) / 2. See doc for simpler solution m = floor(sqrt(n)).
        return int(math.sqrt(param_size))

    @classmethod
    def canon_to_moment(cls, mean: Array, V: Array) -> Array:
        v = V.flatten()
        moment = jnp.concatenate((mean, v))
        return moment

    @classmethod
    def kl(cls, moment1: Array, moment2: Array) -> Scalar:
        m1, V1 = cls.moment_to_canon(moment1)
        m2, V2 = cls.moment_to_canon(moment2)
        return tfp.kl_divergence(
            tfp.MultivariateNormalFullCovariance(m1, V1),
            tfp.MultivariateNormalFullCovariance(m2, V2),
            allow_nan_stats=False,
        )

    @classmethod
    def param_size(cls, state_dim: int) -> int:
        return state_dim + state_dim * state_dim

    @classmethod
    def prior_natural(cls, state_dim) -> Array:
        moment = cls.canon_to_moment(jnp.zeros(state_dim), jnp.eye(state_dim))
        return cls.moment_to_natural(moment)

    @classmethod
    def full_cov(cls, V: Array) -> Array:
        return V

    @classmethod
    def constrain_moment(cls, unconstrained: Array) -> Array:
        n = jnp.size(unconstrained)
        # n = m + m + m
        # m = sqrt(n + 1) - 1
        m = n // 3

        loc, diag, lora = jnp.split(unconstrained, [m, m + m])
        L = jnp.outer(lora, lora)
        V = jnp.diag(constrain_positive(diag)) + L
        v = V.flatten()
        return jnp.concatenate((loc, v))

    @classmethod
    def constrain_natural(cls, unconstrained: Array) -> Array:
        n = jnp.size(unconstrained)
        # n = m + m + m
        # m = sqrt(n + 1) - 1
        m = n // 3

        loc, diag, lora = jnp.split(unconstrained, [m, m + m])
        L = jnp.outer(lora, lora)
        V = jnp.diag(constrain_positive(diag)) + L
        v = -V.flatten()  # negative definite
        return jnp.concatenate((loc, v))

    @classmethod
    def noise_moment(cls, noise_cov) -> Array:
        return jnp.diag(noise_cov)


@registry.register()
class LoRaMVN:
    @classmethod
    def constrain_moment(cls, unconstrained: Array) -> Array:
        n = jnp.size(unconstrained)
        # n = m + 1 + m
        m = (n - 1) // 2

        loc, diag, lora = jnp.split(unconstrained, [m, m + 1])
        L = jnp.outer(lora, lora)
        V = jnp.diag(constrain_positive(diag)) + L
        v = V.flatten()
        return jnp.concatenate((loc, v))

    @classmethod
    def noise_moment(cls, noise_cov) -> Array:
        return jnp.diag(noise_cov)


@registry.register()
class DiagMVN:
    @classmethod
    def natural_to_moment(cls, natural) -> Array:
        nat1, nat2 = jnp.split(natural, 2)
        cov = -0.5 / nat2
        mean = -0.5 * nat1 / nat2
        return jnp.concatenate((mean, cov))

    @classmethod
    def moment_to_natural(cls, moment) -> Array:
        mean, cov = cls.moment_to_canon(moment)
        nat2 = -0.5 / cov
        nat1 = mean / cov
        return jnp.concatenate((nat1, nat2))

    @classmethod
    def sample_by_moment(cls, key, moment, mc_size=None) -> Array:
        mean, cov = cls.moment_to_canon(moment)
        shape = None if mc_size is None else (mc_size,)
        return jrandom.multivariate_normal(
            key, mean, jnp.diag(cov), shape=shape
        )  # It seems JAX does the reparameterization trick

    @classmethod
    def moment_to_canon(cls, moment) -> tuple:
        mean, cov = jnp.split(
            moment, 2, -1
        )  # trick: the 2nd moment here is actually cov diag
        return mean, cov

    @classmethod
    def canon_to_moment(cls, mean, cov) -> Array:
        moment = jnp.concatenate((mean, cov))
        return moment

    @classmethod
    def variable_size(cls, param_size: int) -> int:
        # n: size of vectorized mean param
        # m: size of random variable
        # n = m + m*m
        # m = (sqrt(1 + 4n) - 1) / 2. See doc for simpler solution m = floor(sqrt(n)).
        return param_size // 2

    @classmethod
    def kl(cls, moment1, moment2) -> Scalar:
        m1, cov1 = cls.moment_to_canon(moment1)
        m2, cov2 = cls.moment_to_canon(moment2)
        return tfp.kl_divergence(
            tfp.MultivariateNormalDiag(m1, cov1),
            tfp.MultivariateNormalDiag(m2, cov2),
            allow_nan_stats=False,
        )

    @classmethod
    def param_size(cls, state_dim) -> int:
        return 2 * state_dim

    @classmethod
    def prior_natural(cls, state_dim) -> Array:
        """Return standard normal in natural parameter form"""
        moment = cls.canon_to_moment(jnp.zeros(state_dim), jnp.ones(state_dim))
        return cls.moment_to_natural(moment)

    @classmethod
    def full_cov(cls, cov: Array) -> Array:
        return jnp.diag(cov)

    @classmethod
    def constrain_moment(cls, unconstrained) -> Array:
        loc, v = jnp.split(unconstrained, 2)
        v = constrain_positive(v)
        return jnp.concatenate((loc, v))

    @classmethod
    def constrain_natural(cls, unconstrained) -> Array:
        n1, n2 = jnp.split(unconstrained, 2)
        n2 = -constrain_positive(n2)
        return jnp.concatenate((n1, n2))

    @classmethod
    def unconstrain_natural(cls, natural) -> Array:
        n1, n2 = jnp.split(natural, 2)
        n2 = unconstrain_positive(-n2)
        return jnp.concatenate((n1, n2))

    @classmethod
    def noise_moment(cls, noise_cov) -> Array:
        return noise_cov
