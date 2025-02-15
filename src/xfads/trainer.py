from dataclasses import dataclass
from functools import partial

import jax
from jax import numpy as jnp, random as jrandom
from jaxtyping import Array, Scalar, Float
import optax
import chex
import equinox as eqx
from tqdm import trange

from . import core, vi, smoother


@dataclass
class Opt:
    min_iter: int = 20
    max_iter: int = 20
    learning_rate: float = 1e-3
    clip_norm: float = 5.0
    batch_size: int = 1
    weight_decay: float = 1e-3
    seed: int = 0
    dropout: float = 0.0
    noise_eta: float = 0.5
    noise_gamma: float = 0.8


def make_optimizer(model, opt: Opt):
    # optimizer = optax.chain(
    #     optax.clip_by_global_norm(opt.clip_norm),
    #     optax.adamw(opt.learning_rate),
    # )
    optimizer = optax.chain(
        optax.clip_by_global_norm(opt.clip_norm),
        optax.add_noise(opt.noise_eta, opt.noise_gamma, opt.seed),
        optax.scale_by_adam(),
        optax.add_decayed_weights(opt.weight_decay),
        optax.scale_by_learning_rate(opt.learning_rate),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    return optimizer, opt_state


def loader(
    *arrays, batch_size, key
):  # -> Generator[tuple[Any, Any, KeyArray], Any, None]:
    chex.assert_equal_shape(arrays, dims=0)

    n: int = jnp.size(arrays[0], 0)
    q = n // batch_size
    m = n % batch_size

    key, permkey = jrandom.split(key)
    perm = jax.random.permutation(permkey, jnp.arange(n))

    K = q + 1 if m > 0 else q
    for k in range(K):
        indices = perm[k * batch_size : (k + 1) * batch_size]
        ret = tuple(arr[indices] for arr in arrays)
        key_k = jrandom.fold_in(key, k)
        yield key_k, *ret


def train(
    model: smoother.XFADS,
    data,
    *,
    key: Array,
    opt: Opt,
) -> smoother.XFADS:
    chex.assert_equal_shape(data, dims=(0, 1))

    def batch_elbo(model, key, ts, moment_s, moment_p, ys) -> Array:
        _elbo = jax.vmap(
            jax.vmap(
                partial(
                    vi.elbo,
                    eloglik=model.likelihood.eloglik,
                    approx=model.hyperparam.approx,
                    mc_size=model.hyperparam.mc_size,
                )
            )
        )  # (batch, seq)

        keys = jrandom.split(key, ys.shape[:2])  # ys.shape[:2] + (2,)

        return _elbo(keys, ts, moment_s, moment_p, ys)

    def batch_sample(
        key, moment: Float[Array, "batch time moment"], approx
    ) -> Float[Array, "batch time variable"]:
        def seq_sample(key, moment: Float[Array, "time moment"]):
            keys = jrandom.split(key, jnp.size(moment, 0))
            ret = jax.vmap(approx.sample_by_moment)(keys, moment)
            chex.assert_shape(ret, moment.shape[:1] + (None,))
            return ret

        keys = jrandom.split(key, jnp.size(moment, 0))
        ret = jax.vmap(seq_sample)(keys, moment)
        # jax.debug.print("\nmoment shape={shape}\n", shape=moment.shape)
        chex.assert_shape(ret, moment.shape[:2] + (None,))

        return ret

    def batch_fb_predict(model, z, u):
        ztp1 = eqx.filter_vmap(eqx.filter_vmap(model.forward))(z, u)
        zt = eqx.filter_vmap(eqx.filter_vmap(model.backward))(ztp1, u)
        return zt

    def batch_bf_predict(model, z, u):
        ztm1 = eqx.filter_vmap(eqx.filter_vmap(model.backward))(z, u)
        zt = eqx.filter_vmap(eqx.filter_vmap(model.forward))(ztm1, u)
        return zt

    @eqx.filter_jit
    def batch_loss(model, key, tb, yb, ub, wb) -> Scalar:
        key, subkey = jrandom.split(key)
        chex.assert_equal_shape((tb, yb, ub), dims=(0, 1))

        key, subkey = jrandom.split(key)
        _, moment, moment_p = smoother.batch_smooth(model, subkey, tb, yb, ub, opt.dropout)

        key, subkey = jrandom.split(key)
        zb = batch_sample(subkey, moment, model.hyperparam.approx)
        zb_hat_fb = batch_fb_predict(model, zb, ub)
        zb_hat_bf = batch_bf_predict(model, zb, ub)
        fb_loss = jnp.mean((zb_hat_fb - zb_hat_bf) ** 2)

        key, subkey = jrandom.split(key)
        free_energy = -batch_elbo(model, subkey, tb, moment, moment_p, yb)

        chex.assert_equal_shape((free_energy, wb))
        
        loss = (
            jnp.mean(free_energy)
            + model.hyperparam.fb_penalty * fb_loss
            + model.hyperparam.noise_penalty * model.forward.loss()
            # + hyperparam.noise_penalty * model.backward.loss()
        )  

        return loss

    def optimize(model, key, batch_loss_func, *data):
        optimizer, opt_state = make_optimizer(model, opt)

        @eqx.filter_value_and_grad
        def loss_func(model, key, *data) -> Scalar:
            return batch_loss_func(model, key, *data)

        @eqx.filter_jit
        def step(model, key, opt_state, *data):
            loss, grads = loss_func(model, key, *data)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

        old_loss = jnp.inf
        terminate = False
        with trange(opt.max_iter) as pbar:
            for i in pbar:
                try:
                    key_i = jrandom.fold_in(key, i)
                    for minibatch_key, *minibatch in loader(
                        *data, batch_size=opt.batch_size, key=key_i
                    ):
                        model, opt_state, loss = step(
                            model, minibatch_key, opt_state, *minibatch
                        )
                except KeyboardInterrupt:
                    terminate = True

                key, subkey = jrandom.split(key, 2)
                loss = batch_loss_func(model, subkey, *data)
                pbar.set_postfix({"loss": f"{loss:.3f}"})

                chex.assert_tree_all_finite(loss)

                if terminate:
                    break

                if jnp.isclose(loss, old_loss) and i > opt.min_iter:
                    break

                old_loss = loss

            return model, loss

    model, loss = optimize(model, key, batch_loss, *data)

    return model
