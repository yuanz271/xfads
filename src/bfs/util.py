from dataclasses import InitVar, dataclass, field
from jax import random as jrnd
from jaxtyping import PRNGKeyArray


@dataclass
class Context:
    key: PRNGKeyArray
    state_dim: int
    input_dim: int
    observation_dim: int
    mc_size: int
    
    def newkey(self, n: int=1):
        if n < 1:
            raise ValueError
        newkey, *subkeys = jrnd.split(self.key, n + 1)
        self.key = newkey
        return subkeys
