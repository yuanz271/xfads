from typing import ClassVar, Protocol, Type

from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray
import tensorflow_probability.substrates.jax.distributions as tfp
import equinox as eqx

from .nn import VariantBiasLinear, StationaryLinear
from .constraints import constrain_positive, unconstrain_positive
from .distributions import Approx


MAX_LOGRATE = 7.


class Observation(Protocol):
    def eloglik(self, key: PRNGKeyArray, t: Array, moment: Array, y: Array, approx, mc_size: int): ...

    def set_static(self, static=True): ...

    def initialize(self, *args, **kwargs): ...

    def init(self, *args, **kwargs): ...


class Likelihood(eqx.Module):
    registry: ClassVar[dict] = dict()

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        Likelihood.registry[cls.__name__] = cls


def get_class(name) -> Type:
    return Likelihood.registry[name]


class Poisson(Likelihood):
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
        loglam = eta + 0.5 * cvc
        # loglam = jnp.where(loglam < MAX_LOGRATE, loglam, jnp.log(loglam))
        loglam = jnp.minimum(loglam, MAX_LOGRATE)
        lam = jnp.exp(loglam)
        ll = jnp.sum(y*eta - lam)
        return ll


class DiagGaussian(Likelihood):
    unconstrained_cov: Array = eqx.field(static=False)
    readout: eqx.Module

    def __init__(self, state_dim, observation_dim, *, key, norm_readout: bool = False, **kwargs):
        cov = kwargs.get('cov', jnp.ones(observation_dim))
        self.unconstrained_cov = unconstrain_positive(cov)

        n_steps = kwargs.get('n_steps', 0)

        if n_steps > 0:
            self.readout = VariantBiasLinear(state_dim, observation_dim, n_steps, key=key, norm_readout=norm_readout)
        else:
            self.readout = StationaryLinear(state_dim, observation_dim, key=key, norm_readout=norm_readout)

    def cov(self):
        return constrain_positive(self.unconstrained_cov)

    def eloglik(self, key: PRNGKeyArray, t: Array, moment: Array, y: Array, approx: Type[Approx], mc_size: int) -> Array:
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
