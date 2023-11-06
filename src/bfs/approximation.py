"""
Exponential-family variational distributions
"""
from abc import ABC, abstractclassmethod


class Approximation(ABC):
    """Interface of exponential-family distributions
    The subclasses should implement class functions performing
    converion between natural parameter and mean parameter
    """

    @abstractclassmethod
    def natural_to_mean(cls, natural_params):
        pass

    @abstractclassmethod
    def mean_to_natural(cls, mean_params):
        pass

    # @abstractclassmethod
    # def fisher(cls, natural_params):
    #     pass


class Gaussian(Approximation):
    @classmethod
    def natural_to_mean(cls, natural_params):
        print("This is Gaussian natural to mean")
        # raise NotImplementedError()

    @classmethod
    def mean_to_natural(cls, natural_params):
        print("This is Gaussian mean to natural")
        # raise NotImplementedError()


if __name__ == "__main__":
    # approx = Approximation()  # expect an error

    approx = Gaussian()
    approx.natural_to_mean(None)
    approx.mean_to_natural(None)
