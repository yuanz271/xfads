from equinox import Module
from typing import TypedDict


class Modules(TypedDict):
    dynamics: Module
    state_noise: Module
    likelihood: Module
    obs_encoder: Module
    back_encoder: Module
    covariate: Module


class ModelSpec(TypedDict):
    observation_dim: int
    state_dim: int
    input_dim: int
    width: int
    depth: int
    emission_noise: float
    state_noise: float
    dynamics: str
    approx: str
    norm_readout: bool
    mc_size: int
    random_state: int
    min_iter: int
    max_em_iter: int
    max_inner_iter: int
    batch_size: int
    static_params: str
