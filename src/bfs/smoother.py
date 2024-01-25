from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Type
from jaxtyping import Array, PRNGKeyArray, Scalar
import jax
from jax import numpy as jnp, random as jrandom
import optax
import chex
import equinox as eqx
from equinox import Module, nn as enn
from sklearn.base import TransformerMixin
from tqdm import trange

from . import vi, smoothing
from .dynamics import GaussianStateNoise, Nonlinear
from .vi import DiagGaussainLik, Likelihood
from .distribution import DiagMVN, ExponentialFamily
from .smoothing import Hyperparam


@dataclass
class Opt:
    max_iter: int = 1
    learning_rate: float = 1e-3
    clip_norm: float = 1.0


def batch_smoother(
    dynamics,
    statenoise,
    likelihood,
    obs_encoder,
    back_encoder,
    hyperparam,
) -> Callable[[Array, Array, PRNGKeyArray], tuple[Array, Array]]:
    smooth = jax.vmap(
        partial(
            smoothing.smooth,
            dynamics=dynamics,
            statenoise=statenoise,
            likelihood=likelihood,
            obs_encoder=obs_encoder,
            back_encoder=back_encoder,
            hyperparam=hyperparam,
        )
    )

    def _smooth(ys, us, key) -> tuple[Array, Array]:
        return smooth(ys, us, jrandom.split(key, jnp.size(ys, 0)))

    return _smooth


def batch_elbo(
    eloglik, approx, mc_size
) -> Callable[[PRNGKeyArray, Array, Array, Array, Optional[Callable]], Scalar]:
    elbo = jax.vmap(
        jax.vmap(partial(vi.elbo, eloglik=eloglik, approx=approx, mc_size=mc_size))
    )  # (batch, seq)

    def _elbo(
        key: PRNGKeyArray,
        moment_s: Array,
        moment_p: Array,
        ys: Array,
        *,
        reduce: Callable = jnp.mean,
    ) -> Scalar:
        keys = jrandom.split(key, ys.shape[:2])  # ys.shape[:2] + (2,)
        return reduce(elbo(keys, moment_s, moment_p, ys))

    return _elbo


def make_optimizer(module, opt: Opt):
    optimizer = optax.chain(
        optax.clip_by_global_norm(opt.clip_norm),
        optax.adamw(opt.learning_rate),
    )
    opt_state = optimizer.init(eqx.filter(module, eqx.is_inexact_array))

    return optimizer, opt_state


def train(
    y: Array,
    u: Array,
    dynamics,
    statenoise,
    likelihood: Likelihood,
    obs_encoder,
    back_encoder,
    hyperparam: Hyperparam,
    *,
    key: PRNGKeyArray,
    opt: Opt,
) -> tuple:
    chex.assert_rank((y, u), 3)

    m_modules = (dynamics, likelihood)
    e_modules = (statenoise, obs_encoder, back_encoder)

    opt_m, opt_m_state = make_optimizer(m_modules, opt)
    opt_e, opt_e_state = make_optimizer(e_modules, opt)

    def batch_loss(key, y, u, m_modules, e_modules) -> Scalar:
        dynamics, likelihood = m_modules
        statenoise, obs_encoder, back_encoder = e_modules

        smooth = batch_smoother(
            dynamics,
            statenoise,
            likelihood,
            obs_encoder,
            back_encoder,
            hyperparam,
        )
        elbo = batch_elbo(likelihood.eloglik, hyperparam.approx, hyperparam.mc_size)

        skey, lkey = jrandom.split(key, 2)
        moment_s, moment_p = smooth(y, u, skey)
        return -elbo(lkey, moment_s, moment_p, y)

    def eloss(e_modules, key) -> Scalar:
        return batch_loss(key, y, u, m_modules, e_modules)

    def mloss(m_modules, key) -> Scalar:
        return batch_loss(key, y, u, m_modules, e_modules)

    @eqx.filter_jit
    def step(loss_func, module, optimizer, opt_state, key):
        loss, grads = eqx.filter_value_and_grad(loss_func)(module, key)
        updates, opt_state = optimizer.update(grads, opt_state, module)
        module = eqx.apply_updates(module, updates)
        return module, opt_state, loss

    def train_loop(loss_func, modules, optimizer, opt_state, key):
        old_loss = jnp.inf
        for i in range(opt.max_iter):
            key_i = jrandom.fold_in(key, i)
            modules, opt_state, loss = step(
                loss_func, modules, optimizer, opt_state, key_i
            )
            if jnp.isclose(loss, old_loss):
                break
            old_loss = loss
        return loss, modules, opt_state

    key, subkey = jrandom.split(key)
    old_loss = jnp.inf
    for i in (pbar := trange(opt.max_iter)):
        ekey, mkey = jrandom.split(jrandom.fold_in(subkey, i))
        loss_e, e_modules, opt_e_state = train_loop(
            eloss, e_modules, opt_e, opt_e_state, ekey
        )
        loss_m, m_modules, opt_m_state = train_loop(
            mloss, m_modules, opt_m, opt_m_state, mkey
        )
        loss = 0.5 * (loss_e + loss_m)
        pbar.set_postfix({"loss": f"{loss:.3f}"})
        if jnp.isclose(loss, old_loss):
            break
        old_loss = loss

    return m_modules + e_modules


@dataclass
class XFADS(TransformerMixin):
    hyperparam: Hyperparam
    dynamics: Module
    likelihood: Likelihood
    obs_encoder: Module
    back_encoder: Module
    opt: Opt = field(init=False)

    def __init__(
        self,
        observation_dim,
        state_dim,
        input_dim,
        hidden_size,
        n_layers,
        *,
        approx: Type[ExponentialFamily] = DiagMVN,
        mc_size: int = 10,
        random_state: int = 0,
    ) -> None:
        key: PRNGKeyArray = jrandom.PRNGKey(random_state)
        self.hyperparam = Hyperparam(
            approx, state_dim, input_dim, observation_dim, mc_size
        )
        self.opt = Opt(max_iter=100)
        key, dkey, rkey, okey, bkey = jrandom.split(key, 5)
        self.dynamics = Nonlinear(state_dim, input_dim, hidden_size, n_layers, key=dkey)
        self.statenoise = GaussianStateNoise(jnp.ones(state_dim))
        self.likelihood = DiagGaussainLik(
            cov=jnp.ones(observation_dim),
            readout=enn.Linear(state_dim, observation_dim, key=rkey),
        )
        self.obs_encoder = smoothing.get_obs_encoder(
            state_dim, observation_dim, hidden_size, n_layers, key=okey
        )
        self.back_encoder = smoothing.get_back_encoder(
            state_dim, hidden_size, n_layers, key=bkey
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

        smooth = batch_smoother(
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
