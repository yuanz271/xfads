from functools import partial
from typing import Self, Type
import json

from jaxtyping import Array, PRNGKeyArray
import jax
from jax import numpy as jnp, random as jrandom
import equinox as eqx
import chex

from . import core, distribution, dynamics, observations, spec, encoders
from .core import Hyperparam


def skip_data(x, prob, *, key) -> tuple[PRNGKeyArray, Array, Array]:
    """
    return a key whenever consume a key
    """
    key, subkey = jrandom.split(key)
    mask = jrandom.bernoulli(key, 1 - prob, x.shape[:2] + (1,))  # broadcast to the last dimension
    return subkey, mask


class XFADS(eqx.Module):
    spec: dict = eqx.field(static=True)
    hyperparam: Hyperparam = eqx.field(static=True)
    forward: eqx.Module
    backward: eqx.Module
    likelihood: eqx.Module
    alpha_encoder: eqx.Module
    beta_encoder: eqx.Module

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
        observation: str = "Poisson",
        n_steps: int = 0,
        norm_readout: bool = False,
        mc_size: int = 1,
        random_state: int = 0,
        static_params: str = "",
        fb_penalty: float = 0.0,
        noise_penalty: float = 0.0,
        dyn_kwargs=None,
        obs_kwargs=None,
        enc_kwargs=None,
    ) -> None:
        if dyn_kwargs is None:
            dyn_kwargs = {}

        if obs_kwargs is None:
            obs_kwargs = {}

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
            dyn_kwargs=dyn_kwargs,
            obs_kwargs=obs_kwargs,
            enc_kwargs=enc_kwargs,
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
            mode,
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
            **obs_kwargs,
        )
        
        #####
        key, subkey = jrandom.split(key)
        self.alpha_encoder = encoders.AlphaEncoder(state_dim, observation_dim, approx=approx, key=subkey, **enc_kwargs)
        
        key, subkey = jrandom.split(key)
        self.beta_encoder = encoders.BetaEncoder(state_dim, approx=approx, key=subkey, **enc_kwargs)
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
    
    def init(self, data: tuple):
        T, Y, U = data
        mean_y = jnp.mean(Y, axis=0)
        biases = jnp.log(jnp.maximum(mean_y, 1e-6))
        return eqx.tree_at(lambda model: model.likelihood.readout.biases, self, biases)

    def transform(
        self, data, *, key: Array
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
def batch_smooth(model, key, t, y, u, dropout) -> tuple[Array, Array, Array]:
    batch_alpha_encode = jax.vmap(jax.vmap(model.alpha_encoder))
    batch_constrain_natural = jax.vmap(jax.vmap(model.hyperparam.approx.constrain_natural))
    batch_beta_encode = jax.vmap(model.beta_encoder)

    if model.hyperparam.mode == "pseudo":
        _smooth = jax.vmap(partial(core.filter, model=model))
        def batch_encode(y, key) -> Array:
            mask_y = jnp.all(jnp.isfinite(y), axis=2, keepdims=True)  # nonfinite are missing values
            chex.assert_equal_shape((y, mask_y), dims=(0,1))
            y = jnp.where(mask_y, y, 0)

            key, mask_a = skip_data(y, dropout, key=key)
            key, mask_b = skip_data(y, dropout, key=key)
            key, mask_ab = skip_data(y, dropout, key=key)

            key, subkey = jrandom.split(key)
            a = batch_constrain_natural(batch_alpha_encode(y, key=jrandom.split(subkey, y.shape[:2])))
            a = jnp.where(mask_y, a, 0)  # miss_values have no updates to state
            a = jnp.where(mask_a, a, 0)  # skip random bins
            
            key, subkey = jrandom.split(key)
            b = batch_constrain_natural(batch_beta_encode(a, key=jrandom.split(subkey, jnp.size(a, 0))))
            b = jnp.where(mask_b, b, 0)

            ab = a + b
            ab = jnp.where(mask_ab, ab, 0)

            return ab
    else:
        _smooth = jax.vmap(partial(core.bismooth, model=model))
        def batch_encode(y, key) -> Array:
            valid_mask = jnp.all(jnp.isfinite(y), axis=2, keepdims=True)  # nonfinite are missing values
            chex.assert_equal_shape((y, valid_mask), dims=(0,1))
            y = jnp.where(valid_mask, y, 0)
            
            key, mask_a, _ = skip_data(key, y, dropout)

            a = batch_constrain_natural(batch_alpha_encode(y))
            a = jnp.where(valid_mask, a, 0)

            return jnp.where(mask_a, a, 0)
        
    key, subkey = jrandom.split(key)
    alpha = batch_encode(y, subkey)

    return _smooth(jrandom.split(key, len(t)), t, alpha, u)
