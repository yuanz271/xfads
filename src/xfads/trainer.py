from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import jax
from jax import numpy as jnp, random as jrandom
from tqdm import trange
from jaxtyping import Array, PRNGKeyArray, Scalar
import chex
import optax
import equinox as eqx

from . import smoothing, vi
from .smoothing import Hyperparam
from .vi import Likelihood


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


@dataclass
class Opt:
    max_inner_iter: int = 1
    max_em_iter: int = 1
    learning_rate: float = 1e-3
    clip_norm: float = 5.0
    batch_size: int = 1


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
