from dataclasses import dataclass
from functools import partial

from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
import numpy as np
import jax
from jax import numpy as jnp, random as jrandom
from jaxtyping import Array, Scalar, Float
import optax
import chex
import equinox as eqx
from tqdm import trange

from . import vi, smoother


@dataclass
class Opt:
    min_iter: int = 50
    max_iter: int = 50
    learning_rate: float = 1e-3
    clip_norm: float = 5.0
    batch_size: int = 1
    weight_decay: float = 1e-3
    seed: int = 0
    noise_eta: float = 0.5
    noise_gamma: float = 0.8
    valid_ratio: float = 0.2


class Stopping:
    def __init__(self, initial_loss, min_improvement=0, min_epoch=0, patience=0):
        self.min_improvement = min_improvement
        self.min_epoch = min_epoch
        self.patience = patience
        self._losses = [initial_loss]

    def should_stop(self, loss: float) -> bool:
        stop = False
        self._losses.append(loss)

        if len(self._losses) <= self.min_epoch:
            return False

        average_improvement = -np.mean(
            np.diff(self._losses[-max(self.patience + 1, 1) :])
        )

        if average_improvement < self.min_improvement:
            stop = True

        return stop


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


def data_loader(
    *arrays, batch_size, rng
):  # -> Generator[tuple[Any, Any, KeyArray], Any, None]:
    n: int = np.size(arrays[0], 0)
    perm = rng.permutation(n)
    start = 0
    end = batch_size
    while end <= n and start < n:
        batch_perm = perm[start:end]
        yield tuple(arr[batch_perm] for arr in arrays)  # move minibatch to GPU
        start = end
        end = start + batch_size


def train(
    model: smoother.XFADS,
    data,
    *,
    seed: int,
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

    def batch_fb_predict(model, z, u, *, key):
        fkey, bkey = jrandom.split(key) 
        ztp1 = jax.vmap(jax.vmap(model.forward))(z, u, key=jrandom.split(fkey, z.shape[:2]))
        zt = jax.vmap(jax.vmap(model.backward))(ztp1, u, key=jrandom.split(bkey, z.shape[:2]))
        return zt

    def batch_bf_predict(model, z, u, *, key):
        fkey, bkey = jrandom.split(key) 
        ztm1 = jax.vmap(jax.vmap(model.backward))(z, u, key=jrandom.split(bkey, z.shape[:2]))
        zt = jax.vmap(jax.vmap(model.forward))(ztm1, u, key=jrandom.split(fkey, z.shape[:2]))
        return zt

    def batch_loss(model, key, tb, yb, ub) -> Scalar:
        key, subkey = jrandom.split(key)
        chex.assert_equal_shape((tb, yb, ub), dims=(0, 1))

        key, subkey = jrandom.split(key)
        _, moment, moment_p = model(tb, yb, ub, key=subkey)
        
        if model.hyperparam.mode == "bifilter":
            key, skey, fbkey, bfkey = jrandom.split(key, 4)
            zb = batch_sample(skey, moment, model.hyperparam.approx)
            zb_hat_fb = batch_fb_predict(model, zb, ub, key=fbkey)
            zb_hat_bf = batch_bf_predict(model, zb, ub, key=bfkey)
            fb_loss = jnp.mean((zb_hat_fb - zb_hat_bf) ** 2)
        else:
            fb_loss = 0.

        key, subkey = jrandom.split(key)
        free_energy = -batch_elbo(model, subkey, tb, moment, moment_p, yb)

        loss = (
            jnp.mean(free_energy)
            + model.hyperparam.fb_penalty * fb_loss
            + model.hyperparam.noise_penalty * model.forward.loss()
            # + hyperparam.noise_penalty * model.backward.loss()
        )

        return loss

    def optimize(model, seed, batch_loss_func, *data):
        rng = np.random.default_rng(seed)
        key = jrandom.key(seed)

        n_devices = len(jax.devices())

        n: int = np.size(data[0], 0)
        perm = rng.permutation(n)
        n_samples_per_device = int(n * opt.valid_ratio / n_devices)
        n_valid = n_samples_per_device * n_devices  # n_devices-dividible for sharding

        indices_valid, indices_training = perm[:n_valid], perm[n_valid:]
        valid_set = tuple(arr[indices_valid] for arr in data)
        training_set = tuple(arr[indices_training] for arr in data)

        mesh = jax.make_mesh((n_devices,), ("batch",))

        optimizer, opt_state = make_optimizer(model, opt)
        
        @eqx.filter_jit
        def evaluate(model, key, data):
            model = eqx.nn.inference_mode(model)
            return shard_map(
                lambda *d: batch_loss(model, key, *d),
                mesh,
                in_specs=(
                    P(
                        "batch", None
                    ),
                    P(
                        "batch", None, None
                    ),
                    P(
                        "batch", None, None
                    ),
                ),
                out_specs=P(),
                check_rep=False,
            )(*data)

        @eqx.filter_value_and_grad
        def loss_func(model, key, *data) -> Scalar:
            return batch_loss_func(model, key, *data)

        @eqx.filter_jit
        def step(model, key, opt_state, *data):
            loss, grads = loss_func(model, key, *data)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss
        
        key, eval_key = jrandom.split(key)
        terminate = False
        batch_size = opt.batch_size
        stopping = Stopping(evaluate(model, eval_key, valid_set).item(), 0, opt.min_iter, 5)

        with trange(opt.max_iter) as pbar:
            for i in pbar:
                epoch_key: Array = jrandom.fold_in(key, i)
                try:
                    epoch_key, loader_key = jrandom.split(epoch_key)
                    for j, batch in enumerate(
                        data_loader(*training_set, batch_size=batch_size, rng=rng)
                    ):
                        batch_key = jrandom.fold_in(epoch_key, j)
                        model, opt_state, loss = step(
                            model, batch_key, opt_state, *batch
                        )
                except KeyboardInterrupt:
                    terminate = True

                _, subkey = jrandom.split(epoch_key)
                epoch_loss = evaluate(model, subkey, valid_set).item()
                # epoch_loss /= j+1
                pbar.set_postfix({"loss": f"{epoch_loss:.3f}"})

                chex.assert_tree_all_finite(epoch_loss)

                if terminate or stopping.should_stop(epoch_loss):
                    break

        return model, stopping._losses

    model, losses = optimize(model, seed, batch_loss, *data)

    return model
