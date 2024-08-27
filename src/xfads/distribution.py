"""
Exponential-family variational distributions
"""
import math
from typing import Protocol

from jax import nn as jnn, numpy as jnp, random as jrandom
from jaxtyping import Array, Scalar, PRNGKeyArray
import tensorflow_probability.substrates.jax.distributions as tfp
import chex

from .nn import make_mlp


EPS = 1e-6


def _inv(a):
    return jnp.linalg.inv(a + EPS * jnp.eye(a.shape[-1]))


class ExponentialFamily(Protocol):
    """Interface of exponential-family distributions
    The subclasses should implement class functions performing
    converion between natural parameter and mean parameter.
    The subclasses should be stateless.
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


class MVN(ExponentialFamily):
    @classmethod
    def moment_to_canon(cls, moment: Array) -> tuple[Array, Array]: ...

    @classmethod
    def canon_to_moment(cls, mean: Array, cov: Array) -> Array: ...

    @classmethod
    def full_cov(cls, cov: Array) -> Array: ...

    @classmethod
    def constrain_natural(cls, unconstrained: Array) -> Array: ...
    

class FullMVN(MVN):
    @classmethod
    def natural_to_moment(cls, natural: Array) -> Array:
        """Pmu, -P/2 => mu, P"""
        n = jnp.size(natural)
        m = cls.variable_size(n)
        nat1, nat2 = jnp.split(natural, [m])
        p = -2 * nat2  # vectorized precision
        P = jnp.reshape(p, (m, m))  # precision matrix
        mean = jnp.linalg.solve(P, nat1)
        moment = jnp.concatenate((mean, p))
        return moment

    @classmethod
    def moment_to_natural(cls, moment: Array) -> Array:
        mean, P = cls.moment_to_canon(moment)
        Nat2 = -0.5 * P
        nat2 = Nat2.flatten()
        nat1 = P @ mean
        natural = jnp.concatenate((nat1, nat2))
        return natural

    @classmethod
    def sample_by_moment(cls, key: PRNGKeyArray, moment: Array, mc_size: int) -> Array:
        mean, P = cls.moment_to_canon(moment)
        cov = _inv(P)
        return jrandom.multivariate_normal(key, mean, cov, shape=(mc_size,))  # It seems JAX does reparameterization trick

    @classmethod
    def moment_to_canon(cls, moment: Array) -> tuple[Array, Array]:
        n = jnp.size(moment)
        m = cls.variable_size(n)
        mean, p = jnp.split(moment, [m])
        P = jnp.reshape(p, (m, m))
        return mean, P
    
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
    def canon_to_moment(cls, mean: Array, P: Array) -> Array:
        p = P.flatten()
        moment = jnp.concatenate((mean, p))
        return moment
    
    @classmethod
    def kl(cls, moment1: Array, moment2: Array) -> Scalar:
        m1, P1 = cls.moment_to_canon(moment1)
        m2, P2 = cls.moment_to_canon(moment2)
        return tfp.kl_divergence(tfp.MultivariateNormalFullCovariance(m1, _inv(P1)), tfp.MultivariateNormalFullCovariance(m2, _inv(P2)))

    @classmethod
    def param_size(cls, state_dim: int) -> int:
        return state_dim + state_dim * state_dim

    @classmethod
    def prior_natural(cls, state_dim) -> Array:
        moment = cls.canon_to_moment(jnp.zeros(state_dim), jnp.eye(state_dim))
        return cls.moment_to_natural(moment)

    @classmethod
    def full_cov(cls, P: Array) -> Array:
        return _inv(P)
    
    @classmethod
    def constrain_natural(cls, unconstrained: Array) -> Array:
        n = jnp.size(unconstrained)
        # n = m + m + m
        # m = sqrt(n + 1) - 1
        m = n // 3

        nat1, diag, nat2 = jnp.split(unconstrained, [m, m + m])
        L = jnp.outer(nat2, nat2)
        N2 = - (jnp.diag(jnn.softplus(diag)) + L)  # ND
        nat2 = N2.flatten()
        return jnp.concatenate((nat1, nat2))

    @classmethod
    def get_encoders(cls, observation_dim, state_dim, depth, width, key) -> tuple:
        obs_key, back_key = jrandom.split(key)
        unconstrained_size = state_dim + state_dim + state_dim
        obs_enc = make_mlp(
            observation_dim, unconstrained_size, width, depth, key=obs_key
        )
        back_enc = make_mlp(
            cls.param_size(state_dim) * 2,
            unconstrained_size,
            width,
            depth,
            key=back_key,
        )
        return obs_enc, back_enc


class LoRaMVN(MVN):
    @classmethod
    def natural_to_moment(cls, natural: Array) -> Array:
        """Pmu, -P/2 => mu, P"""
        n = jnp.size(natural)
        m = cls.variable_size(n)
        nat1, nat2 = jnp.split(natural, [m])
        p = -2 * nat2  # vectorized precision
        P = jnp.reshape(p, (m, m))  # precision matrix
        mean = jnp.linalg.solve(P, nat1)
        moment = jnp.concatenate((mean, p))
        return moment

    @classmethod
    def moment_to_natural(cls, moment: Array) -> Array:
        mean, P = cls.moment_to_canon(moment)
        Nat2 = -0.5 * P
        nat2 = Nat2.flatten()
        nat1 = P @ mean
        natural = jnp.concatenate((nat1, nat2))
        return natural

    @classmethod
    def sample_by_moment(cls, key: PRNGKeyArray, moment: Array, mc_size: int) -> Array:
        mean, P = cls.moment_to_canon(moment)
        cov = _inv(P)
        return jrandom.multivariate_normal(key, mean, cov, shape=(mc_size,))  # It seems JAX does reparameterization trick

    @classmethod
    def moment_to_canon(cls, moment: Array) -> tuple[Array, Array]:
        n = jnp.size(moment)
        m = cls.variable_size(n)
        mean, p = jnp.split(moment, [m])
        P = jnp.reshape(p, (m, m))
        return mean, P
    
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
    def canon_to_moment(cls, mean: Array, P: Array) -> Array:
        p = P.flatten()
        moment = jnp.concatenate((mean, p))
        return moment
    
    @classmethod
    def kl(cls, moment1: Array, moment2: Array) -> Scalar:
        m1, P1 = cls.moment_to_canon(moment1)
        m2, P2 = cls.moment_to_canon(moment2)
        return tfp.kl_divergence(tfp.MultivariateNormalFullCovariance(m1, _inv(P1)), tfp.MultivariateNormalFullCovariance(m2, _inv(P2)))

    @classmethod
    def param_size(cls, state_dim: int) -> int:
        return state_dim + state_dim * state_dim

    @classmethod
    def prior_natural(cls, state_dim) -> Array:
        moment = cls.canon_to_moment(jnp.zeros(state_dim), jnp.eye(state_dim))
        return cls.moment_to_natural(moment)

    @classmethod
    def full_cov(cls, P: Array) -> Array:
        return _inv(P)
    
    @classmethod
    def constrain_natural(cls, unconstrained: Array) -> Array:
        n = jnp.size(unconstrained)
        # n = m + 1 + m
        m = (n - 1) // 2

        nat1, diag, nat2 = jnp.split(unconstrained, [m, m + 1])
        L =  jnp.outer(nat2, nat2)
        N2 = - (jnn.softplus(diag) * jnp.eye(m) + L)  # ND
        nat2 = N2.flatten()
        return jnp.concatenate((nat1, nat2))

    @classmethod
    def get_encoders(cls, observation_dim, state_dim, depth, width, key) -> tuple:
        obs_key, back_key = jrandom.split(key)
        unconstrained_size = state_dim + 1 + state_dim
        obs_enc = make_mlp(
            observation_dim, unconstrained_size, width, depth, key=obs_key
        )
        back_enc = make_mlp(
            cls.param_size(state_dim) * 2,
            unconstrained_size,
            width,
            depth,
            key=back_key,
        )
        return obs_enc, back_enc


class DiagMVN(MVN):
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
    def sample_by_moment(cls, key, moment, mc_size) -> Array:
        mean, cov = cls.moment_to_canon(moment)
        return jrandom.multivariate_normal(
            key, mean, jnp.diag(cov), shape=(mc_size,)
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
            tfp.MultivariateNormalDiag(m1, cov1), tfp.MultivariateNormalDiag(m2, cov2)
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
    def constrain_natural(cls, unconstrained) -> Array:
        nat1, nat2 = jnp.split(unconstrained, 2)
        nat2 = -jnn.softplus(nat2)
        return jnp.concatenate((nat1, nat2))

    @classmethod
    def get_encoders(cls, observation_dim, state_dim, width, depth, key) -> tuple:
        obs_key, back_key = jrandom.split(key)
        obs_enc = make_mlp(
            observation_dim, cls.param_size(state_dim), width, depth, key=obs_key
        )
        back_enc = make_mlp(
            cls.param_size(state_dim) * 2,
            cls.param_size(state_dim),
            width,
            depth,
            key=back_key,
        )
        return obs_enc, back_enc
