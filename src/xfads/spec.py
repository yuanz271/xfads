from equinox import Module
from typing import TypedDict


class Modules(TypedDict):
    dynamics: Module
    state_noise: Module
    likelihood: Module
    obs_encoder: Module
    back_encoder: Module
    covariate: Module


class EncoderSpec(TypedDict):
    depth: int
    width: int


class StateModelSpec(TypedDict):
    module: str
    depth: int
    width: int
    state_noise: float


class NeuralModelSpec(TypedDict):
    emission_noise: float


class OptSpec(TypedDict):
    mode: str
    max_inner_iter: int
    max_outer_iter: int
    learning_rate: float
    clip_norm: float
    batch_size: int
    static: str


class ModelSpec(TypedDict):
    neural_dim: int
    state_dim: int
    input_dim: int
    covariate_dim: int
    approx: str
    mc_size: int
    seed: int
    dyn_spec: StateModelSpec
    lik_spec: NeuralModelSpec    
    enc_spec: EncoderSpec
    opt_spec: OptSpec
