from collections.abc import Callable
from dataclasses import dataclass
import json
from typing import Optional, Self

import jax
from jax import numpy as jnp, random as jrandom
import equinox as eqx
from equinox import nn as enn
from jaxtyping import Array, PRNGKeyArray
from sklearn.base import TransformerMixin
import numpy as np

from . import smoothing, distribution, dynamics, spec, trainer
from .nn import Constant
from .dynamics import Diffusion, DiagMVNStateNoise
from .vi import DiagMVNLik
from .distribution import DiagMVN
from .smoothing import Hyperparam, make_batch_smoother


@dataclass
class XFADS(TransformerMixin):
    hyperparam: Hyperparam
    modules: spec.Modules  # trainable modules
    opt: trainer.Opt
    seed: int

    def __init__(
        self,
        neural_dim,
        state_dim,
        input_dim,
        covariate_dim,
        *,
        approx: str = "DiagMVN",
        mc_size: int = 1,
        seed: Optional[int] = None,
        dyn_spec: spec.StateModelSpec = {},
        lik_spec: spec.NeuralModelSpec = {},
        enc_spec: spec.EncoderSpec = {},
        opt_spec: spec.OptSpec = {},
    ) -> None:
        # Generate JAX key from NumPy RNG
        sq = np.random.SeedSequence(seed)  # can be None
        self.seed = sq.entropy  # save for reproducibility
        key_seed, _ = sq.generate_state(2)

        self.spec: spec.ModelSpec = {
            "neural_dim": neural_dim,
            "state_dim": state_dim,
            "input_dim": input_dim,
            "covariate_dim": covariate_dim,
            "approx": approx,
            "seed": self.seed,
            "mc_size": mc_size,
            "dyn_spec": dyn_spec,
            "enc_spec": enc_spec,
            "lik_spec": lik_spec,
            "opt_spec": opt_spec,
        }

        key: PRNGKeyArray = jrandom.PRNGKey(key_seed)

        approx = getattr(distribution, approx, DiagMVN)

        self.hyperparam = Hyperparam(
            approx, state_dim, input_dim, neural_dim, covariate_dim, mc_size
        )

        self.opt = trainer.Opt(**opt_spec)

        key, dynamics_key, likelihood_key, obs_key, back_key, covariate_key = (
            jrandom.split(key, 6)
        )

        dynamics_class = getattr(dynamics, dyn_spec["module"], Diffusion)

        self.modules = dict(
            dynamics=dynamics_class(
                state_dim, input_dim, key=dynamics_key, kwargs=dyn_spec
            ),
            state_noise=DiagMVNStateNoise(
                jnp.full((state_dim,), fill_value=dyn_spec["state_noise"])
            ),
            likelihood=DiagMVNLik(
                cov=jnp.full((neural_dim,), fill_value=lik_spec["emission_noise"]),
                readout=enn.Linear(state_dim, neural_dim, key=likelihood_key),
            ),
            obs_encoder=smoothing.get_obs_encoder(
                state_dim,
                neural_dim,
                approx,
                key=obs_key,
                **enc_spec,
            ),
            back_encoder=smoothing.get_back_encoder(
                state_dim, approx, key=back_key, **enc_spec
            ),
            covariate=enn.Linear(
                covariate_dim, neural_dim, use_bias=False, key=covariate_key
            )
            if covariate_dim > 0
            else Constant(neural_dim, fill_value=0.0),
        )

        self.modules["state_noise"].set_static("s" in self.opt.static)
        self.modules["likelihood"].set_static("l" in self.opt.static)

    def _check_array(
        self, y: Array, u: Optional[Array], x: Optional[Array]
    ) -> tuple[Array, Array, Array]:
        shape = y.shape[:2]

        if u is None:
            u = jnp.zeros(shape + (self.hyperparam.input_dim,))

        if x is None:
            x = jnp.zeros(shape + (self.hyperparam.covariate_dim,))

        return y, u, x

    def fit(self, y: Array, u: Optional[Array], x: Optional[Array]) -> Self:
        """
        :param y: neural
        :param u: input
        :param x: covariate
        :param mode: training mode, joint or em
        """
        sq = np.random.SeedSequence(self.seed)
        _, fit_seed = sq.generate_state(2)
        key = jrandom.PRNGKey(fit_seed)
        
        if self.opt.mode == "joint":
            train = trainer.train_joint
        else:
            train = trainer.train_em

        self.modules = train(
            *self._check_array(y, u, x),
            self,
            key=key,
            opt=self.opt,
        )

    def transform(
        self, y: Array, u: Optional[Array], x: Optional[Array]
    ) -> tuple[Array, Array]:
        sq = np.random.SeedSequence(self.seed)
        _, fit_seed = sq.generate_state(2)
        key = jrandom.PRNGKey(fit_seed)

        smooth: Callable[[Array, Array, PRNGKeyArray], tuple[Array, Array]] = (
            make_batch_smoother(
                self.modules,
                self.hyperparam,
            )
        )

        covariate_layer = jax.vmap(jax.vmap(self.modules["covariate"]))

        y, u, x = self._check_array(y, u, x)

        return smooth(y - covariate_layer(x), u, key)

    def save_model(self, file) -> None:
        with open(file, "wb") as f:
            spec = json.dumps(self.spec)
            f.write((spec + "\n").encode())
            eqx.tree_serialise_leaves(f, self.modules)

    @classmethod
    def load_model(cls, file) -> Self:
        with open(file, "rb") as f:
            spec = json.loads(f.readline().decode())
            model = XFADS(**spec)
            modules = eqx.tree_deserialise_leaves(f, model.modules)
            model.modules = modules
            return model
