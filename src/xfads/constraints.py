import jax
import equinox as eqx
from jaxtyping import ArrayLike

from .nn import softplus_inverse


class AbstractConstraint(eqx.Module):
    def constrain(self, unconstrained: ArrayLike) -> ArrayLike: ...

    def unconstrain(self, constrained: ArrayLike) -> ArrayLike: ...

    def __call__(self, unconstrained: ArrayLike) -> ArrayLike:
        return self.constrain(unconstrained)
    

class Positivity(AbstractConstraint, strict=True):
    def constrain(self, unconstrained: ArrayLike) -> ArrayLike:
        return jax.nn.softplus(unconstrained)
    
    def unconstrain(self, constrained: ArrayLike) -> ArrayLike:
        return softplus_inverse(constrained)
