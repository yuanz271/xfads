from typing import Type, TypedDict


class OptConfig(TypedDict):
    seed: int
    min_iter: int
    max_iter: int
    learning_rate: float
    clip_norm: float
    batch_size: int
    weight_decay: float
    noise_eta: float
    noise_gamma: float
    valid_ratio: float


class Hyperparams(TypedDict):
    approx: Type
    state_dim: int
    input_dim: int
    observation_dim: int
    # covariate_dim: int
    mc_size: int
    fb_penalty: float
    noise_penalty: float
    mode: str


default_opt_config: OptConfig = dict(
    min_iter=50,
    max_iter=50,
    learning_rate=1e-3,
    clip_norm=5.0,
    batch_size=1,
    weight_decay=1e-3,
    seed=0,
    noise_eta=0.5,
    noise_gamma=0.8,
    valid_ratio=0.2,
)
