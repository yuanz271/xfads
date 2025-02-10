from typing import Callable, Type

from jaxtyping import Array, PRNGKeyArray, Scalar

from .distribution import MVN


def elbo(key: PRNGKeyArray, t: Array, moment: Array, moment_p: Array, y: Array, eloglik: Callable[..., Scalar], approx: Type[MVN], *, mc_size: int) -> Scalar:
    """Single time point"""
    ell: Scalar = eloglik(key, t, moment, y, approx, mc_size)
    kl: Scalar = approx.kl(moment, moment_p)
    
    lval: Scalar = ell - kl
    
    return lval
