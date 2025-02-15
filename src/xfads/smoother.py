from functools import partial
from typing import Self, Type
import json

from jaxtyping import Array
import jax
from jax import numpy as jnp, random as jrandom
import equinox as eqx

from . import core, distribution, dynamics, observations, spec, encoders
from .core import Hyperparam


def batch_dropout(key, y, prob) -> Array:
    mask = jrandom.bernoulli(key, 1 - prob, y.shape[:2] + (1,))  # broadcast to the last dimension
    return mask, mask * y / (1 - prob)


class XFADS(eqx.Module):
    spec: dict = eqx.field(static=True)
    hyperparam: Hyperparam = eqx.field(static=True)
    forward: eqx.Module
    backward: eqx.Module
    likelihood: eqx.Module
    obs_encoder: eqx.Module
    backward_encoder: eqx.Module

    def __init__(
        self,
        observation_dim,
        state_dim,
        input_dim,
        emission_noise,
        state_noise,
        *,
        mode: str = "pseudo",
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
        enc_kwargs=None,
    ) -> None:
        if dyn_kwargs is None:
            dyn_kwargs = {}

        if enc_kwargs is None:
            enc_kwargs = {}

        self.spec: spec.ModelSpec = dict(
            mode=mode,
            observation_dim=observation_dim,
            state_dim=state_dim,
            input_dim=input_dim,
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
            enc_kwargs=enc_kwargs,
            dyn_kwargs=dyn_kwargs,
        )

        key = jrandom.key(random_state)

        approx: Type = getattr(distribution, approx, distribution.DiagMVN)

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

        key, fkey, bkey, rkey = jrandom.split(key, 4)

        dynamics_class = dynamics.get_class(forward_dynamics)
        self.forward = dynamics_class(
            key=fkey,
            state_dim=state_dim,
            input_dim=input_dim,
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
        
        #####
        key, subkey = jrandom.split(key)
        self.obs_encoder = encoders.LocalEncoder(state_dim, observation_dim, approx=approx, key=subkey, **enc_kwargs)
        
        key, subkey = jrandom.split(key)
        self.backward_encoder = encoders.BackwardEncoder(state_dim, approx=approx, key=subkey, **enc_kwargs)
        #####

        dynamics_class = dynamics.get_class(backward_dynamics)
        self.backward = dynamics_class(
            key=bkey,
            state_dim=state_dim,
            input_dim=input_dim,
            cov=state_noise,
            **dyn_kwargs,
        )

        if "l" in static_params:
            self.likelihood.set_static()
        if "s" in static_params:
            self.forward.set_static()
    
    def init(self, data: tuple[Array]):
        T, Y, U = data
        mean_y = jnp.mean(Y, axis=0)
        biases = jnp.log(jnp.maximum(mean_y, 1e-6))
        return eqx.tree_at(lambda l: l.likelihood.readout.biases, self, biases)

    def transform(
        self, data: tuple[Array], *, key: Array
    ) -> tuple[Array, Array]:
        _, moment_s, moment_p = batch_smooth(self, key, *data, 0.)
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


@eqx.filter_jit
def batch_smooth(model, key, t, y, u, dropout):
    batch_constrain_natural = jax.vmap(jax.vmap(model.hyperparam.approx.constrain_natural))
    if model.hyperparam.mode == "pseudo":
        _smooth = jax.vmap(partial(core.filter, model=model))
        def batch_encode(x) -> Array:
            a = jax.vmap(jax.vmap(model.obs_encoder))(x)
            b = batch_constrain_natural(jax.vmap(model.backward_encoder)(a))
            # a = batch_constrain_natural(a)
            return b
    else:
        _smooth = jax.vmap(partial(core.bismooth, model=model))
        batch_encode = jax.vmap(jax.vmap(lambda x: model.hyperparam.approx.constrain_natural(model.obs_encoder(x))))

    key, subkey = jrandom.split(key)

    mask, y = batch_dropout(subkey, y, dropout)
    alpha = batch_encode(y)
    alpha = mask * alpha

    return _smooth(jrandom.split(key, len(t)), t, alpha, u)
