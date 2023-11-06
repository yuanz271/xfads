from dataclasses import dataclass
from typing import Callable
from numpy import ndarray
# from sklearn._typing import ArrayLike, MatrixLike
from sklearn.base import TransformerMixin
from jaxtyping import Array, Float

from .approximation import Gaussian


@dataclass
class DBFS(TransformerMixin):
    approximate: type = Gaussian
    dynamics: Callable[[Float[Array, " size"]], Float[Array, " size"]] = None

    def fit_transform(self, X, y=None, **fit_params) -> ndarray:
        raise NotImplementedError()

    def optimize_natural(
        self,
        natural_params: Float[Array, " time natural_size"],
        predicted_natural_params: Float[Array, " time natural_size"],
        natural_update_from_observation,
        #  mean_params: Float[Array, " time mean_size"],
        multiplier,
        fisher_inverse,
        *,
        max_iter,
        alpha,
    ):
        M = None
        for i in range(max_iter):
            mean_params = self.approximate.natural_to_mean(natural_params)
            predicted_mean_params = self.approximate.natural_to_mean(
                predicted_natural_params
            )
            natural_params = (1 - alpha) * natural_params + alpha * (
                predicted_natural_params
                + natural_update_from_observation
                + M @ multiplier
            )
            predicted_natural_params = predicted_natural_params + alpha * (
                fisher_inverse @ (mean_params - predicted_mean_params) - multiplier
            )
        else:
            "reached last iteration"

        return natural_params, predicted_natural_params
