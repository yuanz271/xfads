from functools import partial
from typing import Self, Type, Callable
import json

from jaxtyping import Array, PRNGKeyArray
import jax
from jax import numpy as jnp, random as jrandom
import equinox as eqx
import chex

from . import core, distributions, dynamics, observations, encoders
from .core import Hyperparam, Mode


class DataMasker(eqx.Module, strict=True):
    p: float
    inference: bool

    def __init__(
        self,
        p: float = 0.5,
        inference: bool = False,
    ):

        self.p = p
        self.inference = inference

    @jax.named_scope("xfads.DataMasker")
    def __call__(
        self,
        x: Array,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Array:
        
        if inference is None:
            inference = self.inference

        if isinstance(self.p, (int, float)) and self.p == 0:
            inference = True
        
        shape = x.shape[:2] + (1,)  # broadcast to the last dimension
        if inference:
            return key, jnp.ones(shape)
        elif key is None:
            raise RuntimeError(
                f"{self.__name__} requires a key when running in non-deterministic mode."
            )
        else:
            key, subkey = jrandom.split(key)
            q  = 1 - jax.lax.stop_gradient(self.p)
            mask = jrandom.bernoulli(key, q, shape) 
            return subkey, mask


class XFADS(eqx.Module, strict=True):
    spec: dict = eqx.field(static=True)
    hyperparam: Hyperparam = eqx.field(static=True)
    forward: eqx.Module
    backward: eqx.Module
    likelihood: eqx.Module
    alpha_encoder: eqx.Module
    beta_encoder: eqx.Module
    masker: eqx.Module
    unconstrained_prior_natural: Array

    def __init__(
        self,
        observation_dim,
        state_dim,
        input_dim,
        state_noise,
        *,
        mode: str,
        forward_dynamics: str,
        backward_dynamics: str,
        approx: str,
        observation: str,
        n_steps: int,
        mc_size: int,
        seed: int,
        fb_penalty: float,
        noise_penalty: float,
        dropout: float| None,
        dyn_kwargs: dict,
        obs_kwargs: dict,
        enc_kwargs: dict,
    ) -> None:
        self.spec = dict(
            observation_dim=observation_dim,
            state_dim=state_dim,
            input_dim=input_dim,
            state_noise=state_noise,
            mode=mode,
            forward_dynamics=forward_dynamics,
            backward_dynamics=backward_dynamics,
            approx=approx,
            observation=observation,
            n_steps=n_steps,
            mc_size=mc_size,
            seed=seed,
            fb_penalty=fb_penalty,
            noise_penalty=noise_penalty,
            dropout=dropout,
            dyn_kwargs=dyn_kwargs,
            obs_kwargs=obs_kwargs,
            enc_kwargs=enc_kwargs,
        )

        key = jrandom.key(seed) 

        self.masker: DataMasker = DataMasker(dropout)

        approx: Type[distributions.Approx] = distributions.get_class(approx)

        self.hyperparam = Hyperparam(
            approx=approx,
            state_dim=state_dim,
            input_dim=input_dim,
            observation_dim=observation_dim,
            mc_size=mc_size,
            fb_penalty=fb_penalty,
            noise_penalty=noise_penalty,
            mode=mode,
        )

        dynamics_class = dynamics.get_class(forward_dynamics)
        key, subkey = jrandom.split(key)
        self.forward = dynamics_class(
            key=subkey,
            state_dim=state_dim,
            input_dim=input_dim,
            cov=state_noise,
            **dyn_kwargs,
        )
        
        if backward_dynamics is not None:
            dynamics_class = dynamics.get_class(backward_dynamics)
            key, subkey = jrandom.split(key)
            self.backward = dynamics_class(
                key=subkey,
                state_dim=state_dim,
                input_dim=input_dim,
                cov=state_noise,
                **dyn_kwargs,
            )
        else:
            self.backward = None

        observation_class = observations.get_class(observation)
        key, subkey = jrandom.split(key)
        self.likelihood = observation_class(
            state_dim,
            observation_dim,
            key=subkey,
            n_steps=n_steps,
            **obs_kwargs,
        )
        
        #####
        key, subkey = jrandom.split(key)
        self.alpha_encoder = encoders.AlphaEncoder(state_dim, observation_dim, approx=approx, key=subkey, **enc_kwargs)
        
        key, subkey = jrandom.split(key)
        self.beta_encoder = encoders.BetaEncoder(state_dim, approx=approx, key=subkey, **enc_kwargs)
        #####

        # if "l" in static_params:
        #     self.likelihood.set_static()
        # if "s" in static_params:
        #     self.forward.set_static()

        self.unconstrained_prior_natural = approx.unconstrain_natural(approx.prior_natural(state_dim))
    
    def init(self, t, y, u):
        mean_y = jnp.mean(y, axis=0)
        biases = jnp.log(jnp.maximum(mean_y, 1e-6))
        return eqx.tree_at(lambda model: model.likelihood.readout.biases, self, biases)
    
    def prior_natural(self) -> Array:
        return self.hyperparam.approx.constrain_natural(self.unconstrained_prior_natural)

    def save_model(self, file) -> None:
        with open(file, "wb") as f:
            spec = json.dumps(self.spec)
            f.write((spec + "\n").encode())
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def load_model(cls, file):
        with open(file, "rb") as f:
            spec = json.loads(f.readline().decode())
            model = XFADS(**spec)
            return eqx.tree_deserialise_leaves(f, model)
    
    @eqx.filter_jit
    def __call__(self, t, y, u, *, key) -> tuple[Array, Array, Array]:
        batch_alpha_encode: Callable = jax.vmap(jax.vmap(self.alpha_encoder))
        batch_constrain_natural: Callable = jax.vmap(jax.vmap(self.hyperparam.approx.constrain_natural))
        batch_beta_encode: Callable = jax.vmap(self.beta_encoder)

        match self.hyperparam.mode:
            case Mode.BIFILTER:
                batch_smooth = jax.vmap(partial(core.bismooth, model=self))
                def batch_encode(y, key) -> Array:
                    mask_y = jnp.all(jnp.isfinite(y), axis=2, keepdims=True)  # nonfinite are missing values
                    # chex.assert_equal_shape((y, valid_mask), dims=(0,1))
                    y = jnp.where(mask_y, y, 0)

                    key, subkey = jrandom.split(key)
                    a = batch_constrain_natural(batch_alpha_encode(y, key=jrandom.split(subkey, y.shape[:2])))
                    a = jnp.where(mask_y, a, 0)

                    key, mask_a= self.masker(a, key=key)
                    a = jnp.where(mask_a, a, 0)

                    return a
            case _:
                batch_smooth = jax.vmap(partial(core.filter, model=self))
                def batch_encode(y, key) -> Array:
                    # handling missing values
                    mask_y = jnp.all(jnp.isfinite(y), axis=2, keepdims=True)  # nonfinite are missing values
                    # chex.assert_equal_shape((y, mask_y), dims=(0, 1))
                    y = jnp.where(mask_y, y, 0)
                    
                    key, subkey = jrandom.split(key)
                    a = batch_constrain_natural(batch_alpha_encode(y, key=jrandom.split(subkey, y.shape[:2])))
                    a = jnp.where(mask_y, a, 0)  # miss_values have no updates to state

                    key, mask_a = self.masker(y, key=key)
                    a = jnp.where(mask_a, a, 0)  # pseudo missing values
        
                    b = batch_constrain_natural(batch_beta_encode(a, key=jrandom.split(key, a.shape[0])))
                    key, mask_b = self.masker(y, key=key)
                    b = jnp.where(mask_b, b, 0)  # filter only steps

                    ab = a + b

                    # key, mask_ab = self.masker(y, key=key)
                    # ab = jnp.where(mask_ab, ab, 0)

                    return ab
                
        key, subkey = jrandom.split(key)
        alpha = batch_encode(y, subkey)

        return batch_smooth(jrandom.split(key, len(t)), t, alpha, u)
