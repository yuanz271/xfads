"""
Approximate distributions
"""
from abc import ABCMeta, abstractmethod
from jax import numpy as jnp, random as jrandom, nn as jnn
from jaxtyping import Array, Scalar, PRNGKeyArray
import tensorflow_probability.substrates.jax.distributions as tfp


class ExponentialFamily(metaclass=ABCMeta):
    """Interface of exponential-family distributions
    The subclasses should implement class functions performing
    converion between natural parameter and mean parameter.
    The subclasses should be stateless.
    """

    @classmethod
    @abstractmethod
    def natural_to_moment(cls, natural: Array) -> Array:
        pass

    @classmethod
    @abstractmethod
    def moment_to_natural(cls, moment: Array) -> Array:
        pass

    @classmethod
    @abstractmethod
    def sample_by_moment(cls, key: PRNGKeyArray, moment: Array, mc_size: int) -> Array:
        pass

    @classmethod
    @abstractmethod
    def moment_size(cls, state_dim: int) -> int:
        pass
    
    @classmethod
    @abstractmethod
    def kl(cls, moment1: Array, moment2: Array) -> Scalar:
        pass

    @classmethod
    @abstractmethod
    def constrain_natural(cls, unconstrained: Array) -> Array:
        pass


class MVNMixin:
    @classmethod
    @abstractmethod
    def moment_to_canon(cls, moment: Array) -> tuple[Array, Array]:
        pass

    @classmethod
    @abstractmethod
    def canon_to_moment(cls, mean: Array, cov: Array) -> Array:
        pass

    @classmethod
    @abstractmethod
    def full_cov(cls, cov: Array) -> Array:
        pass


class MVN(ExponentialFamily, MVNMixin):
    @classmethod
    def natural_to_moment(cls, natural: Array):
        n = jnp.size(natural, -1)
        m = cls.variable_size(n)
        nat1, nat2 = jnp.split(natural, [m], axis=-1)
        precision = -2 * nat2
        mean = jnp.linalg.solve(precision, nat1)
        cov = jnp.linalg.inv(precision)
        moment = cls.canon_to_moment(mean, cov)
        return moment

    @classmethod
    def moment_to_natural(cls, moment: Array):
        mean, cov = cls.moment_to_canon(moment)
        precision = jnp.linalg.inv(cov)
        nat2 = -0.5 * precision
        nat1 = jnp.linalg.solve(cov, mean)
        natural = jnp.concatenate((nat1, nat2), axis=-1)
        return natural

    @classmethod
    def sample_by_moment(cls, key: PRNGKeyArray, moment: Array, mc_size: int) -> Array:
        mean, cov = cls.moment_to_canon(moment)
        return jrandom.multivariate_normal(key, mean, cov, shape=(mc_size,))  # It seems JAX does reparameterization trick

    @classmethod
    def moment_to_canon(cls, moment: Array) -> tuple[Array, Array]:
        n = jnp.size(moment, -1)
        m = cls.variable_size(n)
        mean, mmpcov = jnp.split(moment, [m], axis=-1)
        mmpcov = jnp.reshape(mmpcov, (m, m))
        cov = mmpcov - jnp.outer(mean, mean)
        return mean, cov
    
    @classmethod
    def variable_size(cls, moment_size: int) -> int:
        # n: size of vectorized mean param
        # m: size of random variable
        # n = m + m*m
        # m = (sqrt(1 + 4n) - 1) / 2. See doc for simpler solution m = floor(sqrt(n)).
        return int(jnp.sqrt(moment_size))

    @classmethod
    def canon_to_moment(cls, mean: Array, cov: Array) -> Array:
        mmpcov = jnp.outer(mean, mean) + cov
        mmpcov = mmpcov.flatten()
        moment = jnp.concatenate((mean, mmpcov), axis=-1)
        return moment
    
    @classmethod
    def kl(cls, moment1: Array, moment2: Array) -> Scalar:
        m1, cov1 = cls.moment_to_canon(moment1)
        m2, cov2 = cls.moment_to_canon(moment2)
        return tfp.kl_divergence(tfp.MultivariateNormalFullCovariance(m1, cov1), tfp.MultivariateNormalFullCovariance(m2, cov2))

    @classmethod
    def moment_size(cls, state_dim: int) -> int:
        return state_dim + state_dim * state_dim

    @classmethod
    def full_cov(cls, cov: Array) -> Array:
        return cov
    
    @classmethod
    def constrain_natural(cls, unconstrained: Array) -> Array:
        n = jnp.size(unconstrained, -1)
        m = cls.variable_size(n)
        nat1, nat2 = jnp.split(unconstrained, [m], axis=-1)
        nat2 = jnp.reshape(nat2, (m, m))
        nat2 = - nat2 @ nat2.mT
        nat2 = nat2.flatten()
        return jnp.concatenate((nat1, nat2), axis=-1)


class DiagMVN(ExponentialFamily, MVNMixin):
    @classmethod
    def natural_to_moment(cls, natural: Array) -> Array:
        nat1, nat2 = jnp.split(natural, 2, -1)
        cov = -0.5 / nat2
        mean = -0.5 * nat1 / nat2
        return jnp.concatenate((mean, cov), axis=-1)

    @classmethod
    def moment_to_natural(cls, moment: Array) -> Array:
        mean, cov = cls.moment_to_canon(moment)
        nat2 = -0.5 / cov
        nat1 = mean / cov
        return jnp.concatenate((nat1, nat2), axis=-1)

    @classmethod
    def sample_by_moment(cls, key: PRNGKeyArray, moment: Array, mc_size: int) -> Array:
        mean, cov = cls.moment_to_canon(moment)
        return jrandom.multivariate_normal(key, mean, jnp.diag(cov), shape=(mc_size,))  # It seems JAX does the reparameterization trick

    @classmethod
    def moment_to_canon(cls, moment: Array) -> tuple[Array, Array]:
        mean, cov = jnp.split(moment, 2, -1)  # trick: the 2nd moment here is actually cov diag
        return mean, cov

    @classmethod
    def canon_to_moment(cls, mean: Array, cov: Array) -> Array:
        moment = jnp.concatenate((mean, cov), axis=-1)
        return moment
    
    @classmethod
    def kl(cls, moment1: Array, moment2: Array) -> Scalar:
        m1, cov1 = cls.moment_to_canon(moment1)
        m2, cov2 = cls.moment_to_canon(moment2)
        return tfp.kl_divergence(tfp.MultivariateNormalDiag(m1, cov1), tfp.MultivariateNormalDiag(m2, cov2))
    
    @classmethod
    def moment_size(cls, state_dim: int) -> int:
        return 2 * state_dim

    @classmethod
    def full_cov(cls, cov: Array) -> Array:
        return jnp.diag(cov)

    @classmethod
    def constrain_natural(cls, unconstrained: Array) -> Array:
        nat1, nat2 = jnp.split(unconstrained, 2, axis=-1)
        nat2 = -jnn.softplus(nat2)
        return jnp.concatenate((nat1, nat2), axis=-1)
