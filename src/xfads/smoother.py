from functools import partial
from typing import Self
import json

from jaxtyping import Array
import jax
from jax import numpy as jnp, random as jrandom
import equinox as eqx

from . import core, distribution, dynamics, observations, spec
from .core import Hyperparam
from .trainer import Opt, train




@dataclass
class XFADS(TransformerMixin):
    hyperparam: Hyperparam
    dynamics: Module
    likelihood: Likelihood
    obs_to_update: Module
    back_encoder: Module
    opt: Opt

    def __init__(
        self,
        observation_dim,
        state_dim,
        input_dim,
        width,
        depth,
        emission_noise,
        state_noise,
        *,
        covariate_dim: int = 0,
        forward_dynamics: str = "Nonlinear",
        backward_dynamics: str = "Nonlinear",
        approx: str = "DiagMVN",
        observation: str = "gaussian",
        n_steps: int = 1,
        biases: Array = "none",
        norm_readout: bool = False,
        mc_size: int = 1,
        random_state: int = 0,
        min_iter: int = 0,
        max_em_iter: int = 1,
        max_inner_iter: int = 1,
        batch_size: int = 1,
        static_params: str = "",
        regular: float = 1.,
        dyn_kwargs=None,
    ) -> None:
        if dyn_kwargs is None:
            dyn_kwargs = {}
            
        self.spec: spec.ModelSpec = dict(
            observation_dim=observation_dim,
            state_dim=state_dim,
            input_dim=input_dim,
            width=width,
            depth=depth,
            emission_noise=emission_noise,
            state_noise=state_noise,
            covariate_dim=covariate_dim,
            forward_dynamics=forward_dynamics,
            backward_dynamics=backward_dynamics,
            approx=approx,
            observation=observation,
            n_steps=n_steps,
            biases="none",
            norm_readout=norm_readout,
            mc_size=mc_size,
            random_state=random_state,
            min_iter=min_iter,
            max_em_iter=max_em_iter,
            max_inner_iter=max_inner_iter,
            batch_size=batch_size,
            static_params=static_params,
            regular=regular,
            dyn_kwargs=dyn_kwargs,
        )

        key = jrandom.key(random_state)
        approx = getattr(distribution, approx, distribution.DiagMVN)

        self.hyperparam = Hyperparam(
            approx, state_dim, input_dim, observation_dim, covariate_dim, mc_size, regular
        )
        self.opt = Opt(
            min_iter=min_iter,
            max_em_iter=max_em_iter,
            max_inner_iter=max_inner_iter,
            batch_size=batch_size,
        )

        key, fkey, bkey, rkey, enc_key = jrandom.split(key, 5)
        
        dynamics_class = Registry.get_class(forward_dynamics)
        self.dynamics = dynamics_class(
            key=fkey,
            state_dim=state_dim,
            input_dim=input_dim,
            width=width,
            depth=depth,
            cov=state_noise * jnp.ones(state_dim),
            **dyn_kwargs,
            )
        
        likelihood_class = getattr(vi, observation)
        # print(likelihood_class.__name__)
        if likelihood_class.__name__ == "PoissonLik":
            self.likelihood = PoissonLik(state_dim, observation_dim, key=rkey, norm_readout=norm_readout, n_steps=n_steps, biases=biases)
        else:
            self.likelihood = DiagMVNLik(
                cov=emission_noise * jnp.ones(observation_dim),
                readout=enn.Linear(state_dim, observation_dim, key=rkey),
                norm_readout=norm_readout,
            )

        self.obs_to_update, self.back_encoder = approx.get_encoders(
            observation_dim, state_dim, input_dim, depth, width, enc_key
        )

        # NOTE: temporary
        dynamics_class = Registry.get_class(backward_dynamics)
        self.back_encoder = dynamics_class(
            key=bkey,
            state_dim=state_dim,
            input_dim=input_dim,
            width=width,
            depth=depth,
            cov=state_noise * jnp.ones(state_dim),
            **dyn_kwargs,
            )
        # self.back_encoder = Nonlinear(
        #     key=bkey,
        #     state_dim=state_dim,
        #     input_dim=input_dim,
        #     width=width,
        #     depth=depth,
        #     cov=state_noise * jnp.ones(state_dim),
        #     # **dyn_kwargs,
        #     )

        if "l" in static_params:
            self.likelihood.set_static()
        if "s" in static_params:
            self.dynamics.set_static()

    def fit(self, X: tuple[Array, Array, Array], *, key, mode="joint") -> None:
        match mode:
            # case "em":
            #     _train = train_em
            # case "joint":
            #     _train = partial(train_pseudo, mode=mode)
            # case "pseudo":
            #     _train = partial(train_pseudo, mode=mode)
            case "bi":
                _train = partial(train_pseudo, mode=mode)
            case _:
                raise ValueError(f"Unknown {mode=}")

        t, y, u = X

        (
            self.dynamics,
            self.likelihood,
            self.obs_to_update,
            self.back_encoder,
        ) = _train(
            self.modules(),
            t,
            y,
            u,
            self.hyperparam,
            key=key,
            opt=self.opt,
        )

    def transform(
        self, X: tuple[Array, Array, Array], *, key: Array, mode: str
    ) -> tuple[Array, Array]:
        if mode == "pseudo":
            batch_smooth = jax.vmap(partial(core.smooth_pseudo, hyperparam=self.hyperparam), in_axes=(None, 0, 0, 0, 0))
        elif mode == "bi":
            batch_smooth = jax.vmap(partial(core.bismooth, hyperparam=self.hyperparam), in_axes=(None, 0, 0, 0, 0))
        else:
            batch_smooth = jax.vmap(partial(core.smooth, hyperparam=self.hyperparam), in_axes=(None, 0, 0, 0, 0))

        keys = jrandom.split(key, len(X[0]))
        
        _, moment_s, moment_p = batch_smooth(self.modules(), keys, *X)
        return moment_s, moment_p

    def modules(self):
        return (
            self.dynamics,
            self.likelihood,
            self.obs_to_update,
            self.back_encoder,
        )

    def set_modules(self, modules):
        (
            self.dynamics,
            self.likelihood,
            self.obs_to_update,
            self.back_encoder,
        ) = modules

    def save_model(self, file) -> None:
        with open(file, "wb") as f:
            spec = json.dumps(self.spec)
            f.write((spec + "\n").encode())
            eqx.tree_serialise_leaves(f, self.modules())

    @classmethod
    def load_model(cls, file) -> Self:
        with open(file, "rb") as f:
            spec = json.loads(f.readline().decode())
            model = XFADS(**spec)
            modules = eqx.tree_deserialise_leaves(f, model.modules())
            model.set_modules(modules)
            return model
