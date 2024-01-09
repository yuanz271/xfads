"""
Exponential-family variational distributions
"""
from abc import ABCMeta, abstractmethod
import jax.random as jrnd
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfp


class ExponentialFamily(metaclass=ABCMeta):
    """Interface of exponential-family distributions
    The subclasses should implement class functions performing
    converion between natural parameter and mean parameter.
    The subclasses should be stateless.
    """

    @classmethod
    @abstractmethod
    def natural_to_moment(cls, natural):
        pass

    @classmethod
    @abstractmethod
    def moment_to_natural(cls, moment):
        pass

    @classmethod
    @abstractmethod
    def sample_by_moment(cls, key, moment, mc_size):
        pass

    @classmethod
    @abstractmethod
    def moment_size(cls, state_dim):
        pass


class MVN(ExponentialFamily):
    @classmethod
    def natural_to_moment(cls, natural):
        print("This is Gaussian natural to mean")
        # raise NotImplementedError()

    @classmethod
    def moment_to_natural(cls, moment):
        print("This is Gaussian mean to natural")
        # raise NotImplementedError()

    @classmethod
    def sample_by_moment(cls, key, moment, mc_size):
        mean, cov = cls.moment_to_canon(moment)
        return jrnd.multivariate_normal(key, mean, cov, shape=(mc_size,))  # It seems JAX does reparameterization trick

    @classmethod
    def moment_to_canon(cls, moment):
        # n: size of vectorized mean param
        # m: size of random variable
        # n = m + m*m
        # m = (sqrt(1 + 4n) - 1) / 2. See doc for simpler solution m = floor(sqrt(n)).
        n = jnp.size(moment, 0)
        m = int(jnp.sqrt(n))
        mean, mmpcov = jnp.split(moment, [m])
        mmpcov = jnp.reshape(mmpcov, (m, m))
        cov = mmpcov - jnp.outer(mean, mean)
        return mean, cov

    @classmethod
    def canon_to_moment(cls, mean, cov):
        mmpcov = jnp.outer(mean, mean) + cov
        mmpcov = mmpcov.flatten()
        moment = jnp.concatenate((mean, mmpcov), axis=-1)
        return moment
    
    @classmethod
    def kl(cls, moment1, moment2):
        m1, cov1 = cls.moment_to_canon(moment1)
        m2, cov2 = cls.moment_to_canon(moment2)
        return tfp.kl_divergence(tfp.MultivariateNormalFullCovariance(m1, cov1), tfp.MultivariateNormalFullCovariance(m2, cov2))

    @classmethod
    def moment_size(cls, state_dim):
        return state_dim + state_dim * state_dim


class DiagMVN(ExponentialFamily):
    @classmethod
    def natural_to_moment(cls, natural):
        nat1, nat2 = jnp.split(natural, 2)
        cov = -0.5 / nat2
        mean = -0.5 * nat1 / nat2
        return jnp.concatenate((mean, cov), axis=-1)

    @classmethod
    def moment_to_natural(cls, moment):
        mean, cov = cls.moment_to_canon(moment)
        nat2 = -0.5 / cov
        nat1 = mean / cov
        return jnp.concatenate((nat1, nat2), axis=-1)

    @classmethod
    def sample_by_moment(cls, key, moment, mc_size):
        mean, cov = cls.moment_to_canon(moment)
        return jrnd.multivariate_normal(key, mean, jnp.diag(cov), shape=(mc_size,))  # It seems JAX does reparameterization trick

    @classmethod
    def moment_to_canon(cls, moment):
        mean, cov = jnp.split(moment, 2)  # trick: the 2nd moment here is actually cov diag
        return mean, cov

    @classmethod
    def canon_to_moment(cls, mean, cov):
        moment = jnp.concatenate((mean, cov), axis=-1)
        return moment
    
    @classmethod
    def kl(cls, moment1, moment2):
        m1, cov1 = cls.moment_to_canon(moment1)
        m2, cov2 = cls.moment_to_canon(moment2)
        return tfp.kl_divergence(tfp.MultivariateNormalDiag(m1, cov1), tfp.MultivariateNormalDiag(m2, cov2))
    
    @classmethod
    def moment_size(cls, state_dim):
        return 2 * state_dim
