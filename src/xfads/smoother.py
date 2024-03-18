from dataclasses import dataclass, field
import json

from jaxtyping import Array, PRNGKeyArray
from jax import numpy as jnp, random as jrandom
import equinox as eqx
from equinox import Module, nn as enn
from sklearn.base import TransformerMixin


from . import smoothing, distribution, dynamics
from .trainer import make_batch_smoother, train, Opt
from .dynamics import Diffusion, GaussianStateNoise
from .vi import DiagGaussainLik, Likelihood
from .distribution import DiagMVN
from .smoothing import Hyperparam


@dataclass
class XFADS(TransformerMixin):
    hyperparam: Hyperparam
    dynamics: Module
    statenoise: Module
    likelihood: Likelihood
    obs_encoder: Module
    back_encoder: Module
    opt: Opt = field(init=False)

    def __init__(
        self,
        observation_dim,
        state_dim,
        input_dim,
        emission_noise,
        state_noise,
        *,
        approx: str = "DiagMVN",
        dyn_mod: str = "Diffusion",
        mc_size: int = 1,
        random_state: int = 0,
        max_em_iter: int = 1,
        max_inner_iter: int = 1,
        batch_size: int = 1,
        enc_kwargs: dict = {},
        dyn_kwargs: dict = {},
        static_params: str = "",
    ) -> None:
        key: PRNGKeyArray = jrandom.PRNGKey(random_state)
        # if isinstance(approx, str) and approx == "DiagMVN":
        #     approx = DiagMVN
        
        approx = getattr(distribution, approx, DiagMVN)

        self.hyperparam = Hyperparam(
            approx, state_dim, input_dim, observation_dim, mc_size
        )
        self.opt = Opt(max_em_iter=max_em_iter, max_inner_iter=max_inner_iter, batch_size=batch_size)
        key, dkey, rkey, okey, bkey = jrandom.split(key, 5)
        
        dynamics_class = getattr(dynamics, dyn_mod, Diffusion)
        self.dynamics = dynamics_class(state_dim, input_dim, key=dkey, kwargs=dyn_kwargs)

        self.statenoise = GaussianStateNoise(state_noise * jnp.ones(state_dim))
        self.likelihood = DiagGaussainLik(
            cov=emission_noise * jnp.ones(observation_dim),
            readout=enn.Linear(state_dim, observation_dim, key=rkey),
        )

        self.statenoise.set_static('s' in static_params)
        self.likelihood.set_static('l' in static_params)

        self.obs_encoder = smoothing.get_obs_encoder(
            state_dim, observation_dim, enc_kwargs['width'], enc_kwargs['depth'], approx, key=okey
        )
        self.back_encoder = smoothing.get_back_encoder(
            state_dim, enc_kwargs['width'], enc_kwargs['depth'], approx, key=bkey
        )

    def fit(self, X: tuple[Array, Array], *, key: PRNGKeyArray) -> None:
        y, u = X
        (
            self.dynamics,
            self.likelihood,
            self.statenoise,
            self.obs_encoder,
            self.back_encoder,
        ) = train(
            y,
            u,
            self.dynamics,
            self.statenoise,
            self.likelihood,
            self.obs_encoder,
            self.back_encoder,
            self.hyperparam,
            key=key,
            opt=self.opt,
        )

    def transform(
        self, X: tuple[Array, Array], *, key: PRNGKeyArray
    ) -> tuple[Array, Array]:
        y, u = X

        smooth = make_batch_smoother(
            self.dynamics,
            self.statenoise,
            self.likelihood,
            self.obs_encoder,
            self.back_encoder,
            self.hyperparam,
        )
        return smooth(y, u, key)
    
    def modules(self):
        return self.dynamics, self.statenoise, self.likelihood, self.obs_encoder, self.back_encoder
    
    def set_modules(self, modules):
        self.dynamics, self.statenoise, self.likelihood, self.obs_encoder, self.back_encoder = modules


def save_model(file, spec, model: XFADS):
    with open(file, "wb") as f:
        spec = json.dumps(spec)
        f.write((spec + "\n").encode())
        eqx.tree_serialise_leaves(f, model.modules())


def load_model(file) -> XFADS:
    with open(file, "rb") as f:
        kwargs = json.loads(f.readline().decode())
        model = XFADS(**kwargs)
        modules = eqx.tree_deserialise_leaves(f, model.modules())
        model.set_modules(modules)
        return model
