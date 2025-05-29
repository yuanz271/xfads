from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Self

from jaxtyping import Array
from jax import numpy as jnp, random as jrnd, vmap
import equinox as eqx
from gearax.modules import ConfModule, load_model, save_model

from . import core, encoders
from .distributions import Approx
from .dynamics import AbstractDynamics
from .observations import Likelihood
from .nn import DataMasker
from .core import Hyperparam, Mode


class XFADS(ConfModule):
    hyperparam: Hyperparam = eqx.field(init=False, static=True)
    forward: eqx.Module = eqx.field(init=False)
    # backward: eqx.Module | None = eqx.field(init=False)
    likelihood: eqx.Module = eqx.field(init=False)
    alpha_encoder: eqx.Module = eqx.field(init=False)
    beta_encoder: eqx.Module = eqx.field(init=False)
    masker: DataMasker = eqx.field(init=False)
    unconstrained_prior_natural: Array = eqx.field(init=False)

    def __post_init__(self, key):
        state_dim = self.conf.state_dim
        # observation_dim = self.conf.observation_dim
        mc_size = self.conf.mc_size
        seed = self.conf.seed
        dropout = self.conf.dropout
        mode = self.conf.mode
        forward = self.conf.forward
        observation = self.conf.observation
        fb_penalty = self.conf.fb_penalty
        noise_penalty = self.conf.noise_penalty

        key = jrnd.key(seed)

        self.masker: DataMasker = DataMasker(dropout)

        approx: type[Approx] = Approx.get_subclass(self.conf.approx)

        self.hyperparam = Hyperparam(
            approx=approx,
            state_dim=state_dim,
            # iu_dim=iu_dim,
            # eu_dim=eu_dim,
            # observation_dim=observation_dim,
            mc_size=mc_size,
            fb_penalty=fb_penalty,
            noise_penalty=noise_penalty,
            mode=mode,
        )

        key, ky = jrnd.split(key)
        self.forward = AbstractDynamics.get_subclass(forward)(
            self.conf.dyn_conf,
            key=ky,
            # state_dim=state_dim,
            # iu_dim=iu_dim,
            # eu_dim=eu_dim,
            # cov=state_noise,
            # **dyn_kwargs,
        )

        # if backward is not None:
        #     key, ky = jrnd.split(key)
        #     self.backward = AbstractDynamics.get_subclass(backward)(
        #         key=ky,
        #         state_dim=state_dim,
        #         iu_dim=iu_dim,
        #         eu_dim=eu_dim,
        #         cov=state_noise,
        #         **dyn_kwargs,
        #     )
        # else:
        #     self.backward = None

        key, ky = jrnd.split(key)
        self.likelihood = Likelihood.get_subclass(observation)(
            self.conf.obs_conf,
            key=ky,
            # state_dim,  # type: ignore
            # observation_dim,
            # n_steps=n_steps,
            # **obs_kwargs,
        )

        #####
        key, ky = jrnd.split(key)
        self.alpha_encoder = encoders.AlphaEncoder(
            self.conf.enc_conf,
            ky,
            # state_dim, observation_dim, approx, key=ky, **enc_kwargs
        )

        key, ky = jrnd.split(key)
        self.beta_encoder = encoders.BetaEncoder(
            self.conf.enc_conf,
            ky,
            # state_dim, approx, key=ky, **enc_kwargs
        )
        #####

        # if "l" in static_params:
        #     self.likelihood.set_static()
        # if "s" in static_params:
        #     self.forward.set_static()

        self.unconstrained_prior_natural = approx.unconstrain_natural(
            approx.prior_natural(state_dim)
        )

    def init(self, t, y, u, c):
        mean_y = jnp.mean(y, axis=0)
        biases = jnp.log(jnp.maximum(mean_y, 1e-6))
        return eqx.tree_at(lambda model: model.likelihood.readout.biases, self, biases)

    @classmethod
    def load(cls, path: str | Path):
        return load_model(path, cls)

    @classmethod
    def save(cls, model: Self, path: str | Path):
        save_model(path, model)

    def prior_natural(self) -> Array:
        return self.hyperparam.approx.constrain_natural(
            self.unconstrained_prior_natural
        )

    def __call__(self, t, y, u, c, *, key) -> tuple[Array, Array, Array]:
        batch_alpha_encode: Callable = vmap(vmap(self.alpha_encoder))  # type: ignore
        batch_constrain_natural: Callable = vmap(
            vmap(self.hyperparam.approx.constrain_natural)
        )
        batch_beta_encode: Callable = vmap(self.beta_encoder)  # type: ignore

        match self.hyperparam.mode:
            case Mode.BIFILTER:
                raise NotImplementedError("BIFILTER mode is not implemented.")
                # batch_smooth = vmap(partial(core.bismooth, model=self))

                # def batch_encode(y: Array, key) -> Array:
                #     mask_y = jnp.all(
                #         jnp.isfinite(y), axis=2, keepdims=True
                #     )  # nonfinite are missing values
                #     # chex.assert_equal_shape((y, valid_mask), dims=(0,1))
                #     y = jnp.where(mask_y, y, 0)

                #     key, ky = jrnd.split(key)
                #     a = batch_constrain_natural(
                #         batch_alpha_encode(y, key=jrnd.split(ky, y.shape[:2]))
                #     )
                #     a: Array = jnp.where(mask_y, a, 0)

                #     key, mask_a = self.masker(a, key=key)
                #     a = jnp.where(mask_a, a, 0)  # type: ignore

                #     return a
            case _:
                batch_smooth = vmap(partial(core.filter, model=self))

                def batch_encode(y: Array, key) -> Array:
                    # handling missing values
                    mask_y = jnp.all(
                        jnp.isfinite(y), axis=2, keepdims=True
                    )  # nonfinite are missing values
                    # chex.assert_equal_shape((y, mask_y), dims=(0, 1))
                    y = jnp.where(mask_y, y, 0)

                    key, ky = jrnd.split(key)
                    a = batch_constrain_natural(
                        batch_alpha_encode(y, key=jrnd.split(ky, y.shape[:2]))
                    )
                    a = jnp.where(mask_y, a, 0)  # miss_values have no updates to state

                    key, mask_a = self.masker(y, key=key)
                    a = jnp.where(mask_a, a, 0)  # type: ignore # pseudo missing values

                    b = batch_constrain_natural(
                        batch_beta_encode(a, key=jrnd.split(key, a.shape[0]))  # type: ignore
                    )
                    key, mask_b = self.masker(y, key=key)
                    b = jnp.where(mask_b, b, 0)  # filter only steps

                    ab = a + b

                    # key, mask_ab = self.masker(y, key=key)
                    # ab = jnp.where(mask_ab, ab, 0)

                    return ab

        key, ky = jrnd.split(key)
        alpha = batch_encode(y, ky)

        return batch_smooth(jrnd.split(key, len(t)), t, alpha, u, c)
