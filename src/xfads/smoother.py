from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Type
import json

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
from .dynamics import GaussianStateNoise, Nonlinear, Linear
from .vi import DiagGaussainLik, Likelihood
from .distribution import DiagMVN, ExponentialFamily
from .smoothing import Hyperparam


@dataclass
class Opt:
    max_inner_iter: int = 1
    max_em_iter: int = 1
    learning_rate: float = 1e-3
    clip_norm: float = 5.0
    batch_size: int = 1


def make_batch_smoother(
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
    
    @eqx.filter_jit
    def _smooth(ys, us, key) -> tuple[Array, Array]:
        return smooth(ys, us, jrandom.split(key, jnp.size(ys, 0)))

    return _smooth


def make_batch_elbo(
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


def loader(ys, us, batch_size, *, key):# -> Generator[tuple[Any, Any, KeyArray], Any, None]:
    chex.assert_equal_shape((ys, us), dims=0)
    
    n: int = jnp.size(ys, 0)
    q = n // batch_size
    m = n % batch_size
    
    key, permkey = jrandom.split(key)
    perm = jax.random.permutation(permkey, jnp.arange(n))

    K = q + 1 if m > 0 else q
    for k in range(K):
        indices = perm[k * batch_size : (k + 1) * batch_size]
        yield ys[indices], us[indices], jrandom.fold_in(key, k)


def batch_loss(m_modules, e_modules, y, u, key, hyperparam) -> Scalar:
    dynamics, likelihood, statenoise, obs_encoder, back_encoder = m_modules + e_modules

    smooth = make_batch_smoother(
        dynamics,
        statenoise,
        likelihood,
        obs_encoder,
        back_encoder,
        hyperparam,
    )
    elbo = make_batch_elbo(likelihood.eloglik, hyperparam.approx, hyperparam.mc_size)

    skey, lkey = jrandom.split(key)
    moment_s, moment_p = smooth(y, u, skey)
    return -elbo(lkey, moment_s, moment_p, y)


def train_loop(modules, y, u, key, step, opt_state, opt):
    old_loss = jnp.inf
    for i in range(opt.max_inner_iter):
        key_i = jrandom.fold_in(key, i)
        total_loss = 0.
        n_minibatch = 0
        for ym, um, minibatch_key in loader(y, u, opt.batch_size, key=key_i):
            modules, opt_state, loss = step(
                modules, ym, um, minibatch_key, opt_state
            )
            # chex.assert_tree_all_finite(loss)
            total_loss += loss
            n_minibatch += 1
        total_loss = total_loss / max(n_minibatch, 1)

        if jnp.isclose(total_loss, old_loss):
            break

        old_loss = total_loss
    return total_loss, modules, opt_state


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

    m_modules = (dynamics, likelihood, statenoise)
    e_modules = (obs_encoder, back_encoder)

    optimizer_m, opt_state_mstep = make_optimizer(m_modules, opt)
    optimizer_e, opt_state_estep = make_optimizer(e_modules, opt)
    
    @eqx.filter_value_and_grad
    def eloss(e_modules, y, u, key) -> Scalar:
        return batch_loss(m_modules, e_modules, y, u, key, hyperparam)
    
    @eqx.filter_value_and_grad
    def mloss(m_modules, y, u, key) -> Scalar:
        return batch_loss(m_modules, e_modules, y, u, key, hyperparam)

    @eqx.filter_jit
    def estep(module, y, u, key, opt_state):
        loss, grads = eloss(module, y, u, key)
        updates, opt_state = optimizer_e.update(grads, opt_state, module)
        module = eqx.apply_updates(module, updates)
        return module, opt_state, loss

    @eqx.filter_jit
    def mstep(module, y, u, key, opt_state):
        loss, grads = mloss(module, y, u, key)
        updates, opt_state = optimizer_m.update(grads, opt_state, module)
        module = eqx.apply_updates(module, updates)
        return module, opt_state, loss

    key, em_key = jrandom.split(key)
    old_loss = jnp.inf
    terminate = False
    for i in (pbar := trange(opt.max_em_iter)):
        try:
            ekey, mkey = jrandom.split(jrandom.fold_in(em_key, i))
            loss_e, e_modules, opt_state_estep = train_loop(
                e_modules, y, u, ekey, estep, opt_state_estep, opt
            )
            loss_m, m_modules, opt_state_mstep = train_loop(
                m_modules, y, u, mkey, mstep, opt_state_mstep, opt
            )
            loss = 0.5 * (loss_e.item() + loss_m.item())

            chex.assert_tree_all_finite(loss)
            pbar.set_postfix({"loss": f"{loss:.3f}"})
        except KeyboardInterrupt:
            terminate = True

        if terminate or jnp.isclose(loss, old_loss):
            break
        old_loss = loss

    return m_modules + e_modules


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
        hidden_size,
        n_layers,
        emission_noise,
        state_noise,
        *,
        approx: Type[ExponentialFamily] = DiagMVN,
        dynamics: str = "linear",
        mc_size: int = 1,
        random_state: int = 0,
        max_em_iter: int = 1,
        max_inner_iter: int = 1,
        batch_size: int = 1,
        enc_kwargs: dict = {},
    ) -> None:
        key: PRNGKeyArray = jrandom.PRNGKey(random_state)
        if isinstance(approx, str) and approx == "DiagMVN":
            approx = DiagMVN
        self.hyperparam = Hyperparam(
            approx, state_dim, input_dim, observation_dim, mc_size
        )
        self.opt = Opt(max_em_iter=max_em_iter, max_inner_iter=max_inner_iter, batch_size=batch_size)
        key, dkey, rkey, okey, bkey = jrandom.split(key, 5)
        if dynamics == "nonlinear":
            self.dynamics = Nonlinear(state_dim, input_dim, hidden_size, n_layers, key=dkey)
        elif dynamics == "linear":
            self.dynamics = Linear(state_dim, input_dim, key=dkey)
        else:
            raise ValueError(f"Unknown dynamics value: {dynamics}")

        self.statenoise = GaussianStateNoise(state_noise * jnp.ones(state_dim))
        self.likelihood = DiagGaussainLik(
            cov=emission_noise*jnp.ones(observation_dim),
            readout=enn.Linear(state_dim, observation_dim, key=rkey),
        )
        self.obs_encoder = smoothing.get_obs_encoder(
            state_dim, observation_dim, enc_kwargs['hidden_size'], enc_kwargs['n_layers'], approx, key=okey
        )
        self.back_encoder = smoothing.get_back_encoder(
            state_dim, enc_kwargs['hidden_size'], enc_kwargs['n_layers'], approx, key=bkey
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
