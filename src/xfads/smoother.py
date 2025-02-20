from functools import partial
from typing import Self, Type, Callable
import json

from jaxtyping import Array, PRNGKeyArray
import jax
from jax import numpy as jnp, random as jrandom
import equinox as eqx
import chex

from . import core, distributions, dynamics, observations, spec, encoders
from .core import Hyperparam


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

    def __init__(
        self,
        observation_dim,
        state_dim,
        input_dim,
        state_noise,
        *,
        mode: str = "pseudo",
        forward_dynamics: str = "Nonlinear",
        backward_dynamics: str = "Nonlinear",
        approx: str = "DiagMVN",
        observation: str = "Poisson",
        n_steps: int = 0,
        mc_size: int = 1,
        random_state: int = 0,
        static_params: str = "",
        fb_penalty: float = 0.0,
        noise_penalty: float = 0.0,
        dyn_kwargs=None,
        obs_kwargs=None,
        enc_kwargs=None,
        dropout: float| None = None,
    ) -> None:
        key = jrandom.key(random_state) 

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
            state_noise=state_noise,
            forward_dynamics=forward_dynamics,
            backward_dynamics=backward_dynamics,
            approx=approx,
            observation=observation,
            n_steps=n_steps,
            mc_size=mc_size,
            random_state=random_state,
            static_params=static_params,
            fb_penalty=fb_penalty,
            noise_penalty=noise_penalty,
            dyn_kwargs=dyn_kwargs,
            obs_kwargs=obs_kwargs,
            enc_kwargs=enc_kwargs,
        )
        self.masker: DataMasker = DataMasker(dropout)

        approx: Type = distributions.get_class(approx)

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

    # def transform(
    #     self, data, *, key: Array
    # ) -> tuple[Array, Array]:
    #     _, moment_s, moment_p = batch_smooth(self, key, *data, 0.)
    #     return moment_s, moment_p

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
    def __call__(self, t, y, u, *, key) -> tuple[Array, Array, Array]:
        batch_alpha_encode: Callable = jax.vmap(jax.vmap(self.alpha_encoder))
        batch_constrain_natural: Callable = jax.vmap(jax.vmap(self.hyperparam.approx.constrain_natural))
        batch_beta_encode: Callable = jax.vmap(self.beta_encoder)

        if self.hyperparam.mode == "pseudo":
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
        else:
            batch_smooth = jax.vmap(partial(core.bismooth, model=self))
            def batch_encode(y, key) -> Array:
                mask_y = jnp.all(jnp.isfinite(y), axis=2, keepdims=True)  # nonfinite are missing values
                # chex.assert_equal_shape((y, valid_mask), dims=(0,1))
                y = jnp.where(mask_y, y, 0)
                a = batch_constrain_natural(batch_alpha_encode(y))
                a = jnp.where(mask_y, a, 0)
                key, mask_a= self.masker(a, key=key)

                return jnp.where(mask_a, a, 0)
            
        key, subkey = jrandom.split(key)
        alpha = batch_encode(y, subkey)

        return batch_smooth(jrandom.split(key, len(t)), t, alpha, u)


# @eqx.filter_jit
# def batch_smooth(model, t, y, u, *, key) -> tuple[Array, Array, Array]:
#     batch_alpha_encode = jax.vmap(jax.vmap(model.alpha_encoder))
#     batch_constrain_natural = jax.vmap(jax.vmap(model.hyperparam.approx.constrain_natural))
#     batch_beta_encode = jax.vmap(model.beta_encoder)

#     if model.hyperparam.mode == "pseudo":
#         _smooth = jax.vmap(partial(core.filter, model=model))
#         def batch_encode(y, key) -> Array:
#             mask_y = jnp.all(jnp.isfinite(y), axis=2, keepdims=True)  # nonfinite are missing values
#             chex.assert_equal_shape((y, mask_y), dims=(0,1))
#             y = jnp.where(mask_y, y, 0)

#             key, mask_a = model.masker(y, key=key)
#             key, mask_b = model.masker(y, key=key)
#             key, mask_ab = model.masker(y, key=key)

#             key, subkey = jrandom.split(key)
#             a = batch_constrain_natural(batch_alpha_encode(y, key=jrandom.split(subkey, y.shape[:2])))
#             a = jnp.where(mask_y, a, 0)  # miss_values have no updates to state
#             a = jnp.where(mask_a, a, 0)  # skip random bins
            
#             key, subkey = jrandom.split(key)
#             b = batch_constrain_natural(batch_beta_encode(a, key=jrandom.split(subkey, jnp.size(a, 0))))
#             b = jnp.where(mask_b, b, 0)

#             ab = a + b
#             ab = jnp.where(mask_ab, ab, 0)

#             return ab
#     else:
#         _smooth = jax.vmap(partial(core.bismooth, model=model))
#         def batch_encode(y, key) -> Array:
#             valid_mask = jnp.all(jnp.isfinite(y), axis=2, keepdims=True)  # nonfinite are missing values
#             chex.assert_equal_shape((y, valid_mask), dims=(0,1))
#             y = jnp.where(valid_mask, y, 0)
            
#             key, mask_a, _ = model.masker(y, key=key)

#             a = batch_constrain_natural(batch_alpha_encode(y))
#             a = jnp.where(valid_mask, a, 0)

#             return jnp.where(mask_a, a, 0)
        
#     key, subkey = jrandom.split(key)
#     alpha = batch_encode(y, subkey)

#     return _smooth(jrandom.split(key, len(t)), t, alpha, u)
