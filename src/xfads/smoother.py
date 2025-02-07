from functools import partial
from typing import Callable, Optional, Self
import json

from jaxtyping import Array, Scalar, Float
import jax
from jax import numpy as jnp, random as jrandom
import optax
import chex
import equinox as eqx
from tqdm import trange

from . import vi, distribution, spec, core
from .dynamics import Registry
from .vi import DiagMVNLik, Likelihood, PoissonLik
from .smoothing import Hyperparam
from .trainer import Opt


# def make_batch_smoother(
#     dynamics,
#     likelihood,
#     obs_to_update,
#     back_encoder,
#     hyperparam,
# ) -> Callable[[Array, Array, Array, Array], tuple[Array, Array]]:
#     smooth = jax.vmap(
#         partial(
#             smoothing.smooth,
#             dynamics=dynamics,
#             likelihood=likelihood,
#             obs_to_update=obs_to_update,
#             back_encoder=back_encoder,
#             hyperparam=hyperparam,
#         )
#     )

#     @eqx.filter_jit
#     def _smooth(ts, ys, us, key) -> tuple[Array, Array]:
#         return smooth(ts, ys, us, jrandom.split(key, jnp.size(ys, 0)))

#     return _smooth


def make_batch_elbo(
    eloglik, approx, mc_size
) -> Callable[[Array, Array, Array, Array, Array, Optional[Callable]], Scalar]:
    elbo = jax.vmap(
        jax.vmap(partial(vi.elbo, eloglik=eloglik, approx=approx, mc_size=mc_size))
    )  # (batch, seq)
    
    @eqx.filter_jit
    def _elbo(
        key: Array,
        ts: Array,
        moment_s: Array,
        moment_p: Array,
        ys: Array,
        *,
        reduce: Callable = jnp.nanmean,
    ) -> Scalar:
        keys = jrandom.split(key, ys.shape[:2])  # ys.shape[:2] + (2,)
        # jax.debug.print("{a}, {b}, {c}, {d}, {e}", a=keys.shape, b=ts.shape, c=moment_s.shape, d=moment_p.shape, e=ys.shape)
        return reduce(elbo(keys, ts, moment_s, moment_p, ys))

    return _elbo


def make_optimizer(module, opt: Opt):
    # optimizer = optax.chain(
    #     optax.clip_by_global_norm(opt.clip_norm),
    #     optax.adamw(opt.learning_rate),
    # )
    optimizer = optax.chain(
        optax.clip_by_global_norm(opt.clip_norm),
        optax.add_noise(1.0, 0.9, opt.seed),
        optax.scale_by_adam(),
        optax.add_decayed_weights(1e-4),
        optax.scale_by_learning_rate(opt.learning_rate),
    )
    opt_state = optimizer.init(eqx.filter(module, eqx.is_inexact_array))

    return optimizer, opt_state


def loader(
    ts, ys, us, batch_size, *, key
):  # -> Generator[tuple[Any, Any, KeyArray], Any, None]:
    chex.assert_equal_shape((ts, ys, us), dims=0)

    n: int = jnp.size(ys, 0)
    q = n // batch_size
    m = n % batch_size

    key, permkey = jrandom.split(key)
    perm = jax.random.permutation(permkey, jnp.arange(n))

    K = q + 1 if m > 0 else q
    for k in range(K):
        indices = perm[k * batch_size : (k + 1) * batch_size]
        yield ts[indices], ys[indices], us[indices], jrandom.fold_in(key, k)


def loader2(
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


# def batch_loss(m_modules, e_modules, t, y, u, key, hyperparam) -> Scalar:
#     dynamics, likelihood, obs_to_update, back_encoder = m_modules + e_modules

#     smooth = make_batch_smoother(
#         dynamics,
#         likelihood,
#         obs_to_update,
#         back_encoder,
#         hyperparam,
#     )
#     elbo = make_batch_elbo(likelihood.eloglik, hyperparam.approx, hyperparam.mc_size)

#     skey, lkey = jrandom.split(key)
#     moment_s, moment_p = smooth(t, y, u, skey)
#     return -elbo(lkey, t, moment_s, moment_p, y)


# def batch_loss_joint(modules, t, y, u, key, hyperparam) -> Scalar:
#     dynamics, likelihood, obs_to_update, back_encoder = modules

#     smooth = make_batch_smoother(
#         dynamics,
#         likelihood,
#         obs_to_update,
#         back_encoder,
#         hyperparam,
#     )
#     elbo = make_batch_elbo(likelihood.eloglik, hyperparam.approx, hyperparam.mc_size)

#     skey, lkey = jrandom.split(key)
#     moment_s, moment_p = smooth(t, y, u, skey)
#     return -elbo(lkey, t, moment_s, moment_p, y)


# def train_loop(modules, t, y, u, key, step, opt_state, opt):
#     # old_loss = jnp.inf
#     for i in range(opt.max_inner_iter):
#         key_i = jrandom.fold_in(key, i)
#         total_loss = 0.0
#         n_minibatch = 0
#         for tm, ym, um, minibatch_key in loader(t, y, u, opt.batch_size, key=key_i):
#             modules, opt_state, loss = step(modules, tm, ym, um, minibatch_key, opt_state)
#             # chex.assert_tree_all_finite(loss)
#             total_loss += loss
#             n_minibatch += 1
#         total_loss = total_loss / max(n_minibatch, 1)

#         # if jnp.isclose(total_loss, old_loss):
#         #     break

#         # old_loss = total_loss
#     return total_loss, modules, opt_state


# def train_loop2(modules, key, step, opt_state, opt, *args):
#     # old_loss = jnp.inf
#     for i in range(opt.max_inner_iter):
#         key_i = jrandom.fold_in(key, i)
#         total_loss = 0.0
#         n_minibatch = 0
#         for tm, ym, um, minibatch_key in loader(*args, opt.batch_size, key=key_i):
#             modules, opt_state, loss = step(modules, tm, ym, um, minibatch_key, opt_state)
#             # chex.assert_tree_all_finite(loss)
#             total_loss += loss
#             n_minibatch += 1
#         total_loss = total_loss / max(n_minibatch, 1)

#         # if jnp.isclose(total_loss, old_loss):
#         #     break

#         # old_loss = total_loss
#     return total_loss, modules, opt_state


# def train_em(
#     t: Array,
#     y: Array,
#     u: Array,
#     dynamics,
#     likelihood: Likelihood,
#     obs_to_update,
#     back_encoder,
#     hyperparam: Hyperparam,
#     *,
#     key: Array,
#     opt: Opt,
# ) -> tuple:
#     chex.assert_rank((y, u), 3)

#     m_modules = (dynamics, likelihood)
#     e_modules = (obs_to_update, back_encoder)

#     optimizer_m, opt_state_mstep = make_optimizer(m_modules, opt)
#     optimizer_e, opt_state_estep = make_optimizer(e_modules, opt)

#     @eqx.filter_value_and_grad
#     def eloss(e_modules, t, y, u, key) -> Scalar:
#         return batch_loss(m_modules, e_modules, t, y, u, key, hyperparam)

#     @eqx.filter_value_and_grad
#     def mloss(m_modules, t, y, u, key) -> Scalar:
#         return batch_loss(m_modules, e_modules, t, y, u, key, hyperparam)

#     @eqx.filter_jit
#     def estep(module, t, y, u, key, opt_state):
#         loss, grads = eloss(module, t, y, u, key)
#         updates, opt_state = optimizer_e.update(grads, opt_state, module)
#         module = eqx.apply_updates(module, updates)
#         return module, opt_state, loss

#     @eqx.filter_jit
#     def mstep(module, t, y, u, key, opt_state):
#         loss, grads = mloss(module, t, y, u, key)
#         updates, opt_state = optimizer_m.update(grads, opt_state, module)
#         module = eqx.apply_updates(module, updates)
#         return module, opt_state, loss

#     key, em_key = jrandom.split(key)
#     old_loss = jnp.inf
#     terminate = False
#     for i in (pbar := trange(opt.max_em_iter)):
#         try:
#             ekey, mkey = jrandom.split(jrandom.fold_in(em_key, i))
#             loss_e, e_modules, opt_state_estep = train_loop(
#                 e_modules, t, y, u, ekey, estep, opt_state_estep, opt
#             )
#             loss_m, m_modules, opt_state_mstep = train_loop(
#                 m_modules, t, y, u, mkey, mstep, opt_state_mstep, opt
#             )
#             loss = 0.5 * (loss_e.item() + loss_m.item())

#             chex.assert_tree_all_finite(loss)
#             pbar.set_postfix({"loss": f"{loss:.3f}"})
#         except KeyboardInterrupt:
#             terminate = True

#         if terminate:
#             break

#         if jnp.isclose(loss, old_loss) and i > opt.min_iter:
#             break

#         old_loss = loss

#     return m_modules + e_modules


def train_pseudo(
    model,
    t: Array,
    y: Array,
    u: Array,
    hyperparam: Hyperparam,
    *,
    key: Array,
    opt: Opt,
    mode: str,
) -> tuple:
    chex.assert_rank((y, u), 3)
    chex.assert_equal_shape((t, y, u), dims=(0, 1))
    
    # dynamics, likelihood, obs_to_update, back_encoder = model.forward, model.likehood
    
    if mode == "pseudo":
        batch_smooth = jax.vmap(partial(core.smooth_pseudo, hyperparam=hyperparam), in_axes=(None, 0, 0, 0, 0))
    elif mode == "bi":
        batch_smooth = jax.vmap(partial(core.bismooth, hyperparam=hyperparam), in_axes=(None, 0, 0, 0, 0))
    else:
        batch_smooth = jax.vmap(partial(core.smooth, hyperparam=hyperparam), in_axes=(None, 0, 0, 0, 0))

    batch_elbo = make_batch_elbo(model.likelihood.eloglik, hyperparam.approx, hyperparam.mc_size)

    def batch_sample(key, moment: Float[Array, "batch time moment"], approx) -> Float[Array, "batch time variable"]:
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

    def batch_loss(model, key, tb, yb, ub) -> Scalar:        
        key, skey, lkey = jrandom.split(key, 3)
        fkeys = jrandom.split(skey, len(tb))
        
        chex.assert_equal_shape((fkeys, tb, yb, ub), dims=0)
        chex.assert_equal_shape((tb, yb, ub), dims=(0, 1))

        _, moment_s, moment_p = batch_smooth(model, fkeys, tb, yb, ub)
        
        zb = batch_sample(key, moment_s, hyperparam.approx)
        zb_hat_fb = batch_fb_predict(model, zb, ub)
        zb_hat_bf = batch_bf_predict(model, zb, ub)
        fb_loss = jnp.mean((zb - zb_hat_fb) ** 2) + jnp.mean((zb - zb_hat_bf) ** 2)
        
        return -batch_elbo(lkey, tb, moment_s, moment_p, yb) + hyperparam.fb_penalty * fb_loss + hyperparam.noise_penalty * (model.forward.loss() + model.backward.loss())

    def _train(model, key, batch_loss_func, *args):

        optimizer, opt_state = make_optimizer(model, opt)

        @eqx.filter_value_and_grad
        def loss_func(model, key, *args) -> Scalar:
            return batch_loss_func(model, key, *args)

        @eqx.filter_jit
        def step(model, key, opt_state, *args):
            loss, grads = loss_func(model, key, *args)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

        old_loss = jnp.inf
        terminate = False
        with trange(opt.max_inner_iter * opt.max_em_iter) as pbar:
            for i in pbar:
                try:
                    key_i = jrandom.fold_in(key, i)
                    for minibatch_key, *argm in loader2(*args, batch_size=opt.batch_size, key=key_i):
                        model, opt_state, loss = step(model, minibatch_key, opt_state, *argm)
                except KeyboardInterrupt:
                    terminate = True
                
                key, subkey = jrandom.split(key, 2)
                loss = batch_loss_func(model, subkey, *args)
                pbar.set_postfix({'loss': f"{loss:.3f}"})

                if terminate:
                    break

                if jnp.isclose(loss, old_loss) and i > opt.min_iter:
                    break

                old_loss = loss

            return model, loss

    model, loss = _train(model, key, batch_loss, t, y, u)

    return model


class XFADS(eqx.Module):
    spec: dict = eqx.field(static=True)
    hyperparam: Hyperparam = eqx.field(static=True)
    opt: Opt = eqx.field(static=True)
    forward: eqx.Module
    backward: eqx.Module
    likelihood: eqx.Module
    obs_encoder: eqx.Module

    def __init__(
        self,
        observation_dim,
        state_dim,
        input_dim,
        width,
        depth,
        emission_noise,
        state_noise,
        *,
        covariate_dim: int = 0,
        forward_dynamics: str = "Nonlinear",
        backward_dynamics: str = "Nonlinear",
        approx: str = "DiagMVN",
        observation: str = "gaussian",
        n_steps: int = 1,
        biases: Array = "none",
        norm_readout: bool = False,
        mc_size: int = 1,
        random_state: int = 0,
        min_iter: int = 0,
        max_em_iter: int = 1,
        max_inner_iter: int = 1,
        batch_size: int = 1,
        static_params: str = "",
        fb_penalty: float = 0.,
        noise_penalty: float = 0., 
        dyn_kwargs=None,
    ) -> None:
        if dyn_kwargs is None:
            dyn_kwargs = {}
            
        self.spec: spec.ModelSpec = dict(
            observation_dim=observation_dim,
            state_dim=state_dim,
            input_dim=input_dim,
            width=width,
            depth=depth,
            emission_noise=emission_noise,
            state_noise=state_noise,
            covariate_dim=covariate_dim,
            forward_dynamics=forward_dynamics,
            backward_dynamics=backward_dynamics,
            approx=approx,
            observation=observation,
            n_steps=n_steps,
            biases="none",
            norm_readout=norm_readout,
            mc_size=mc_size,
            random_state=random_state,
            min_iter=min_iter,
            max_em_iter=max_em_iter,
            max_inner_iter=max_inner_iter,
            batch_size=batch_size,
            static_params=static_params,
            fb_penalty=fb_penalty,
            noise_penalty=noise_penalty,
            dyn_kwargs=dyn_kwargs,
        )

        key = jrandom.key(random_state)
        approx = getattr(distribution, approx, distribution.DiagMVN)

        self.hyperparam = Hyperparam(
            approx, state_dim, input_dim, observation_dim, covariate_dim, mc_size, fb_penalty, noise_penalty
        )
        self.opt = Opt(
            min_iter=min_iter,
            max_em_iter=max_em_iter,
            max_inner_iter=max_inner_iter,
            batch_size=batch_size,
        )

        key, fkey, bkey, rkey, enc_key = jrandom.split(key, 5)
        
        dynamics_class = Registry.get_class(forward_dynamics)
        self.forward = dynamics_class(
            key=fkey,
            state_dim=state_dim,
            input_dim=input_dim,
            width=width,
            depth=depth,
            cov=state_noise * jnp.ones(state_dim),
            **dyn_kwargs,
            )
        
        likelihood_class = getattr(vi, observation)
        # print(likelihood_class.__name__)
        if likelihood_class.__name__ == "PoissonLik":
            self.likelihood = PoissonLik(state_dim, observation_dim, key=rkey, norm_readout=norm_readout, n_steps=n_steps, biases=biases)
        else:
            self.likelihood = DiagMVNLik(
                cov=emission_noise * jnp.ones(observation_dim),
                readout=eqx.nn.Linear(state_dim, observation_dim, key=rkey),
                norm_readout=norm_readout,
            )

        self.obs_encoder, _ = approx.get_encoders(observation_dim, state_dim, input_dim, depth, width, enc_key)

        # NOTE: temporary
        dynamics_class = Registry.get_class(backward_dynamics)
        self.backward = dynamics_class(
            key=bkey,
            state_dim=state_dim,
            input_dim=input_dim,
            width=width,
            depth=depth,
            cov=state_noise * jnp.ones(state_dim),
            **dyn_kwargs,
            )
        # self.back_encoder = Nonlinear(
        #     key=bkey,
        #     state_dim=state_dim,
        #     input_dim=input_dim,
        #     width=width,
        #     depth=depth,
        #     cov=state_noise * jnp.ones(state_dim),
        #     # **dyn_kwargs,
        #     )

        if "l" in static_params:
            self.likelihood.set_static()
        if "s" in static_params:
            self.forward.set_static()

    def fit(self, X: tuple[Array, Array, Array], *, key, mode="joint") -> None:
        match mode:
            # case "em":
            #     _train = train_em
            # case "joint":
            #     _train = partial(train_pseudo, mode=mode)
            # case "pseudo":
            #     _train = partial(train_pseudo, mode=mode)
            case "bi":
                _train = partial(train_pseudo, mode=mode)
            case _:
                raise ValueError(f"Unknown {mode=}")

        t, y, u = X

        return _train(
            self,
            t,
            y,
            u,
            self.hyperparam,
            key=key,
            opt=self.opt,
        )

    def transform(
        self, X: tuple[Array, Array, Array], *, key: Array, mode: str
    ) -> tuple[Array, Array]:
        if mode == "pseudo":
            batch_smooth = jax.vmap(partial(core.smooth_pseudo, hyperparam=self.hyperparam), in_axes=(None, 0, 0, 0, 0))
        elif mode == "bi":
            batch_smooth = jax.vmap(partial(core.bismooth, hyperparam=self.hyperparam), in_axes=(None, 0, 0, 0, 0))
        else:
            batch_smooth = jax.vmap(partial(core.smooth, hyperparam=self.hyperparam), in_axes=(None, 0, 0, 0, 0))

        keys = jrandom.split(key, len(X[0]))
        
        _, moment_s, moment_p = batch_smooth(self, keys, *X)
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
