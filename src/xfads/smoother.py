from functools import partial
from typing import Self
import json

from jaxtyping import Array
import jax
from jax import numpy as jnp, random as jrandom
import equinox as eqx

from . import core, distribution, dynamics, observations, spec
from .core import Hyperparam


def make_batch_smooth(model, dropout):# -> Callable[..., tuple[Any, Any, Array]]:
    _smooth = jax.vmap(partial(core.bismooth, model=model))
    batch_encode = jax.vmap(jax.vmap(lambda x: model.obs_encoder(x)))
    
    @jax.jit
    def batch_smooth(key, t, y, u):
        key, subkey = jrandom.split(key)
        alpha = batch_encode(y)
        return _smooth(jrandom.split(key, len(t)), t, alpha, u)
    
    return batch_smooth


class XFADS(eqx.Module):
    spec: dict = eqx.field(static=True)
    hyperparam: Hyperparam = eqx.field(static=True)
    forward: eqx.Module
    backward: eqx.Module
    likelihood: eqx.Module
    obs_encoder: eqx.Module

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
        n_steps: int = 0,
        norm_readout: bool = False,
        mc_size: int = 1,
        random_state: int = 0,
        static_params: str = "",
        fb_penalty: float = 0.0,
        noise_penalty: float = 0.0,
        dyn_kwargs=None,
        opt_kwargs=None,
    ) -> None:
        if dyn_kwargs is None:
            dyn_kwargs = {}

        if opt_kwargs is None:
            opt_kwargs = {}

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
            norm_readout=norm_readout,
            mc_size=mc_size,
            random_state=random_state,
            static_params=static_params,
            fb_penalty=fb_penalty,
            noise_penalty=noise_penalty,
            dyn_kwargs=dyn_kwargs,
            opt_kwargs=opt_kwargs,
        )

        key = jrandom.key(random_state)
        approx = getattr(distribution, approx, distribution.DiagMVN)

        self.hyperparam = Hyperparam(
            approx,
            state_dim,
            input_dim,
            observation_dim,
            covariate_dim,
            mc_size,
            fb_penalty,
            noise_penalty,
        )
        # self.opt = Opt(**opt_kwargs)

        key, fkey, bkey, rkey, enc_key = jrandom.split(key, 5)

        dynamics_class = dynamics.get_class(forward_dynamics)
        self.forward = dynamics_class(
            key=fkey,
            state_dim=state_dim,
            input_dim=input_dim,
            width=width,
            depth=depth,
            cov=state_noise,
            **dyn_kwargs,
        )

        observation_class = observations.get_class(observation)
        self.likelihood = observation_class(
            state_dim,
            observation_dim,
            key=rkey,
            norm_readout=norm_readout,
            n_steps=n_steps,
        )

        self.obs_encoder = approx.get_encoder(
            observation_dim, state_dim, input_dim, depth, width, enc_key
        )

        dynamics_class = dynamics.get_class(backward_dynamics)
        self.backward = dynamics_class(
            key=bkey,
            state_dim=state_dim,
            input_dim=input_dim,
            width=width,
            depth=depth,
            cov=state_noise,
            **dyn_kwargs,
        )

        if "l" in static_params:
            self.likelihood.set_static()
        if "s" in static_params:
            self.forward.set_static()

    # def fit(self, data: tuple[Array], *, key) -> None:
    #     return trainer.train(
    #         self,
    #         *data,
    #         key=key,
    #     )
    
    def init(self, data: tuple[Array]):
        T, Y, U = data
        mean_y = jnp.mean(Y, axis=0)
        biases = jnp.log(jnp.maximum(mean_y, 1e-6))
        return eqx.tree_at(lambda l: l.likelihood.readout.biases, self, biases)

    def transform(
        self, data: tuple[Array], *, key: Array
    ) -> tuple[Array, Array]:
        batch_smooth = make_batch_smooth(self, dropout=0.)
        _, moment_s, moment_p = batch_smooth(key, *data)
        return moment_s, moment_p

    def save_model(self, file) -> None:
        with open(file, "wb") as f:
            spec = json.dumps(self.spec)
            f.write((spec + "\n").encode())
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def load_model(cls, file) -> Self:
        with open(file, "rb") as f:
            spec = json.loads(f.readline().decode())
            model = XFADS(**spec)
            return eqx.tree_deserialise_leaves(f, model)
