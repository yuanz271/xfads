from collections.abc import Generator
from dataclasses import dataclass

import jax
from jax import numpy as jnp, random as jrandom
from tqdm import trange
from jaxtyping import Array, PRNGKeyArray, Scalar
import chex
import optax
import equinox as eqx

from . import smoothing, vi


@dataclass
class Opt:
    max_inner_iter: int = 1
    max_outer_iter: int = 1
    learning_rate: float = 1e-3
    clip_norm: float = 5.0
    batch_size: int = 1
    static: str = ""


def make_optimizer(module, opt: Opt):
    optimizer = optax.chain(
        optax.clip_by_global_norm(opt.clip_norm),
        optax.adamw(opt.learning_rate),
    )
    opt_state = optimizer.init(eqx.filter(module, eqx.is_inexact_array))

    return optimizer, opt_state


def loader(
    ys, us, xs, *, batch_size, key
) -> Generator[tuple[Array, Array, Array, PRNGKeyArray], None, None]:
    chex.assert_equal_shape((ys, us), dims=0)

    n: int = jnp.size(ys, 0)
    q = n // batch_size
    m = n % batch_size

    key, permkey = jrandom.split(key)
    perm = jax.random.permutation(permkey, jnp.arange(n))

    K = q + 1 if m > 0 else q
    for k in range(K):
        indices = perm[k * batch_size : (k + 1) * batch_size]
        yield ys[indices], us[indices], xs[indices], jrandom.fold_in(key, k)


def batch_loss(modules, y, u, x, key, hyperparam) -> Scalar:
    smooth = smoothing.make_batch_smoother(
        modules,
        hyperparam,
    )
    elbo = vi.make_batch_elbo(
        modules["likelihood"].eloglik, hyperparam.approx, hyperparam.mc_size
    )

    covariate_layer = jax.vmap(jax.vmap(modules["covariate"]))
    covariate_predict = covariate_layer(x)

    skey, lkey = jrandom.split(key)
    moment_s, moment_p = smooth(y - covariate_predict, u, skey)

    return -elbo(lkey, moment_s, moment_p, covariate_predict, y)


# def train_loop(modules, y, u, x, key, step, opt_state, opt):
#     old_loss = jnp.inf
#     for i in range(opt.max_inner_iter):
#         key_i = jrandom.fold_in(key, i)
#         total_loss = 0.
#         n_minibatch = 0
#         for ym, um, minibatch_key in loader(y, u, x, batch_size=opt.batch_size, key=key_i):
#             modules, opt_state, loss = step(
#                 modules, ym, um, minibatch_key, opt_state
#             )
#             # chex.assert_tree_all_finite(loss)
#             total_loss += loss
#             n_minibatch += 1
#         total_loss = total_loss / max(n_minibatch, 1)

#         if jnp.isclose(total_loss, old_loss):
#             break

#         old_loss = total_loss
#     return total_loss, modules, opt_state


# def train_em(
#     y: Array,
#     u: Array,
#     x: Array,
#     model,
#     *,
#     key: PRNGKeyArray,
#     opt: Opt,
# ) -> tuple:
#     chex.assert_rank((y, u, x), 3)

#     (dynamics, statenoise, likelihood, obs_encoder, back_encoder) = model.get_modules()
#     hyperparam = model.hyperparam

#     m_modules = (dynamics, statenoise, likelihood)
#     e_modules = (obs_encoder, back_encoder)

#     optimizer_m, opt_state_mstep = make_optimizer(m_modules, opt)
#     optimizer_e, opt_state_estep = make_optimizer(e_modules, opt)

#     @eqx.filter_value_and_grad
#     def eloss(e_modules, y, u, key) -> Scalar:
#         return batch_loss(m_modules + e_modules, y, u, x, key, hyperparam)

#     @eqx.filter_value_and_grad
#     def mloss(m_modules, y, u, key) -> Scalar:
#         return batch_loss(m_modules + e_modules, y, u, x, key, hyperparam)

#     @eqx.filter_jit
#     def estep(module, y, u, key, opt_state):
#         loss, grads = eloss(module, y, u, key)
#         updates, opt_state = optimizer_e.update(grads, opt_state, module)
#         module = eqx.apply_updates(module, updates)
#         return module, opt_state, loss

#     @eqx.filter_jit
#     def mstep(module, y, u, key, opt_state):
#         loss, grads = mloss(module, y, u, key)
#         updates, opt_state = optimizer_m.update(grads, opt_state, module)
#         module = eqx.apply_updates(module, updates)
#         return module, opt_state, loss

#     key, em_key = jrandom.split(key)
#     old_loss = jnp.inf
#     terminate = False
#     opt.max_inner_iter = 1
#     for i in (pbar := trange(opt.max_em_iter)):
#         try:
#             ekey, mkey = jrandom.split(jrandom.fold_in(em_key, i))
#             loss_e, e_modules, opt_state_estep = train_loop(
#                 e_modules, y, u, ekey, estep, opt_state_estep, opt
#             )
#             loss_m, m_modules, opt_state_mstep = train_loop(
#                 m_modules, y, u, mkey, mstep, opt_state_mstep, opt
#             )
#             loss = 0.5 * (loss_e.item() + loss_m.item())

#             chex.assert_tree_all_finite(loss)
#             pbar.set_postfix({"loss": f"{loss:.3f}"})
#         except KeyboardInterrupt:
#             terminate = True

#         if terminate or jnp.isclose(loss, old_loss):
#             break
#         old_loss = loss

#     return m_modules + e_modules


def train_joint(
    y: Array,
    u: Array,
    x: Array,
    model,
    *,
    key: PRNGKeyArray,
    opt: Opt,
) -> tuple:
    chex.assert_rank((y, u), 3)

    modules = model.modules
    hyperparam = model.hyperparam

    optimizer, opt_state = make_optimizer(modules, opt)

    @eqx.filter_value_and_grad
    def floss(modules, y, u, x, key) -> Scalar:
        return batch_loss(modules, y, u, x, key, hyperparam)

    @eqx.filter_jit
    def step(modules, y, u, x, key, opt_state):
        loss, grads = floss(modules, y, u, x, key)
        updates, opt_state = optimizer.update(grads, opt_state, modules)
        modules = eqx.apply_updates(modules, updates)
        return modules, opt_state, loss

    old_loss: float = jnp.inf
    terminate = False
    for i in (pbar := trange(opt.max_outer_iter)):
        try:
            key_i = jrandom.fold_in(key, i)
            total_loss = 0.0
            n_minibatch = 0
            for ym, um, xm, minibatch_key in loader(
                y, u, x, batch_size=opt.batch_size, key=key_i
            ):
                modules, opt_state, loss = step(
                    modules, ym, um, xm, minibatch_key, opt_state
                )
                total_loss += loss
                n_minibatch += 1

            total_loss: float = total_loss / max(n_minibatch, 1)
            # chex.assert_tree_all_finite(total_loss)
            pbar.set_postfix({"loss": f"{total_loss:.3f}"})
        except KeyboardInterrupt:
            terminate = True

        if terminate or jnp.isclose(total_loss, old_loss):
            break
        old_loss = total_loss

    return modules
