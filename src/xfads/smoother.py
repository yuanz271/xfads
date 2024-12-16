from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Self
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

from . import vi, smoothing, distribution, spec, dynamics as dyn_mod
from .dynamics import GaussianStateNoise
from .vi import DiagMVNLik, Likelihood, NonstationaryPoissonLik, PoissonLik
from .smoothing import Hyperparam


@dataclass
class Opt:
    min_iter: int = 0
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
    def _smooth(ts, ys, us, key) -> tuple[Array, Array]:
        return smooth(ts, ys, us, jrandom.split(key, jnp.size(ys, 0)))

    return _smooth


def make_batch_elbo(
    eloglik, approx, mc_size
) -> Callable[[PRNGKeyArray, Array, Array, Array, Optional[Callable]], Scalar]:
    elbo = jax.vmap(
        jax.vmap(partial(vi.elbo, eloglik=eloglik, approx=approx, mc_size=mc_size))
    )  # (batch, seq)
    
    @eqx.filter_jit
    def _elbo(
        key: PRNGKeyArray,
        ts: Array,
        moment_s: Array,
        moment_p: Array,
        ys: Array,
        *,
        reduce: Callable = jnp.nanmean,
    ) -> Scalar:
        keys = jrandom.split(key, ys.shape[:2])  # ys.shape[:2] + (2,)
        return reduce(elbo(keys, ts, moment_s, moment_p, ys))

    return _elbo


def make_optimizer(module, opt: Opt):
    optimizer = optax.chain(
        optax.clip_by_global_norm(opt.clip_norm),
        optax.adamw(opt.learning_rate),
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


def batch_loss(m_modules, e_modules, t, y, u, key, hyperparam) -> Scalar:
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
    moment_s, moment_p = smooth(t, y, u, skey)
    return -elbo(lkey, t, moment_s, moment_p, y)


def batch_loss_joint(modules, t, y, u, key, hyperparam) -> Scalar:
    dynamics, likelihood, statenoise, obs_encoder, back_encoder = modules

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
    moment_s, moment_p = smooth(t, y, u, skey)
    return -elbo(lkey, t, moment_s, moment_p, y)


def train_loop(modules, t, y, u, key, step, opt_state, opt):
    # old_loss = jnp.inf
    for i in range(opt.max_inner_iter):
        key_i = jrandom.fold_in(key, i)
        total_loss = 0.0
        n_minibatch = 0
        for tm, ym, um, minibatch_key in loader(t, y, u, opt.batch_size, key=key_i):
            modules, opt_state, loss = step(modules, tm, ym, um, minibatch_key, opt_state)
            # chex.assert_tree_all_finite(loss)
            total_loss += loss
            n_minibatch += 1
        total_loss = total_loss / max(n_minibatch, 1)

        # if jnp.isclose(total_loss, old_loss):
        #     break

        # old_loss = total_loss
    return total_loss, modules, opt_state


def train_em(
    t: Array,
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
    def eloss(e_modules, t, y, u, key) -> Scalar:
        return batch_loss(m_modules, e_modules, t, y, u, key, hyperparam)

    @eqx.filter_value_and_grad
    def mloss(m_modules, t, y, u, key) -> Scalar:
        return batch_loss(m_modules, e_modules, t, y, u, key, hyperparam)

    @eqx.filter_jit
    def estep(module, t, y, u, key, opt_state):
        loss, grads = eloss(module, t, y, u, key)
        updates, opt_state = optimizer_e.update(grads, opt_state, module)
        module = eqx.apply_updates(module, updates)
        return module, opt_state, loss

    @eqx.filter_jit
    def mstep(module, t, y, u, key, opt_state):
        loss, grads = mloss(module, t, y, u, key)
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
                e_modules, t, y, u, ekey, estep, opt_state_estep, opt
            )
            loss_m, m_modules, opt_state_mstep = train_loop(
                m_modules, t, y, u, mkey, mstep, opt_state_mstep, opt
            )
            loss = 0.5 * (loss_e.item() + loss_m.item())

            chex.assert_tree_all_finite(loss)
            pbar.set_postfix({"loss": f"{loss:.3f}"})
        except KeyboardInterrupt:
            terminate = True

        if terminate:
            break

        if jnp.isclose(loss, old_loss) and i > opt.min_iter:
            break

        old_loss = loss

    return m_modules + e_modules


def train_joint(
    t: Array,
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

    modules = (dynamics, likelihood, statenoise, obs_encoder, back_encoder)

    optimizer, opt_state_step = make_optimizer(modules, opt)

    @eqx.filter_value_and_grad
    def loss_func(modules, t, y, u, key) -> Scalar:
        return batch_loss_joint(modules, t, y, u, key, hyperparam)

    @eqx.filter_jit
    def joint_step(modules, t, y, u, key, opt_state):
        loss, grads = loss_func(modules, y, u, key)
        updates, opt_state = optimizer.update(grads, opt_state, modules)
        modules = eqx.apply_updates(modules, updates)
        return modules, opt_state, loss

    key, em_key = jrandom.split(key)
    old_loss = jnp.inf
    terminate = False
    max_inner_iter = opt.max_inner_iter
    opt.max_inner_iter = 1
    for i in (pbar := trange(opt.max_em_iter * max_inner_iter)):
        try:
            it_key = jrandom.fold_in(em_key, i)
            loss_joint, modules, opt_state_step = train_loop(
                modules, t, y, u, it_key, joint_step, opt_state_step, opt
            )

            loss = loss_joint.item()

            chex.assert_tree_all_finite(loss)
            pbar.set_postfix({"loss": f"{loss:.3f}"})
        except KeyboardInterrupt:
            terminate = True

        if terminate:
            break

        if jnp.isclose(loss, old_loss) and i > opt.min_iter:
            break

        old_loss = loss

    opt.max_inner_iter = max_inner_iter  # hack
    return modules


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
        width,
        depth,
        emission_noise,
        state_noise,
        *,
        covariate_dim: int = 0,
        dynamics: str = "Nonlinear",
        approx: str = "DiagMVN",
        observation: str = "gaussian",
        n_steps: int = 1,
        norm_readout: bool = False,
        mc_size: int = 1,
        random_state: int = 0,
        min_iter: int = 0,
        max_em_iter: int = 1,
        max_inner_iter: int = 1,
        batch_size: int = 1,
        static_params: str = "",
    ) -> None:
        self.spec: spec.ModelSpec = dict(
            observation_dim=observation_dim,
            state_dim=state_dim,
            input_dim=input_dim,
            width=width,
            depth=depth,
            emission_noise=emission_noise,
            state_noise=state_noise,
            covariate_dim=covariate_dim,
            dynamics=dynamics,
            approx=approx,
            observation=observation,
            n_steps=n_steps,
            norm_readout=norm_readout,
            mc_size=mc_size,
            random_state=random_state,
            min_iter=min_iter,
            max_em_iter=max_em_iter,
            max_inner_iter=max_inner_iter,
            batch_size=batch_size,
            static_params=static_params,
        )

        key: PRNGKeyArray = jrandom.PRNGKey(random_state)
        approx = getattr(distribution, approx, distribution.DiagMVN)

        self.hyperparam = Hyperparam(
            approx, state_dim, input_dim, observation_dim, covariate_dim, mc_size
        )
        self.opt = Opt(
            min_iter=min_iter,
            max_em_iter=max_em_iter,
            max_inner_iter=max_inner_iter,
            batch_size=batch_size,
        )

        key, dkey, rkey, enc_key = jrandom.split(key, 4)
        
        dynamics_class = getattr(dyn_mod, dynamics)
        self.dynamics = dynamics_class(
            key=dkey,
            state_dim=state_dim,
            input_dim=input_dim,
            width=width,
            depth=depth
            )
        self.statenoise = GaussianStateNoise(state_noise * jnp.ones(state_dim))

        if observation == "poisson":
            self.likelihood = PoissonLik(
                readout=enn.Linear(state_dim, observation_dim, key=rkey),
                norm_readout=norm_readout)
        elif observation == "NonstationaryPoissonLik":
            self.likelihood = NonstationaryPoissonLik(state_dim, observation_dim, n_steps, key=rkey, norm_readout=norm_readout)
        else:
            self.likelihood = DiagMVNLik(
                cov=emission_noise * jnp.ones(observation_dim),
                readout=enn.Linear(state_dim, observation_dim, key=rkey),
                norm_readout=norm_readout,
            )

        self.obs_encoder, self.back_encoder = approx.get_encoders(
            observation_dim, state_dim, depth, width, enc_key
        )

        if "l" in static_params:
            self.likelihood.set_static()
        if "s" in static_params:
            self.statenoise.set_static()

    def fit(self, X: tuple[Array, Array, Array], *, key: PRNGKeyArray, mode="em") -> None:
        match mode:
            case "em":
                _train = train_em
            case "joint":
                _train = train_joint
            case _:
                raise ValueError(f"{mode=}")

        t, y, u = X

        (
            self.dynamics,
            self.likelihood,
            self.statenoise,
            self.obs_encoder,
            self.back_encoder,
        ) = _train(
            t,
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
        self, X: tuple[Array, Array, Array], *, key: PRNGKeyArray
    ) -> tuple[Array, Array]:
        t, y, u = X

        smooth = make_batch_smoother(
            self.dynamics,
            self.statenoise,
            self.likelihood,
            self.obs_encoder,
            self.back_encoder,
            self.hyperparam,
        )
        return smooth(t, y, u, key)

    def modules(self):
        return (
            self.dynamics,
            self.statenoise,
            self.likelihood,
            self.obs_encoder,
            self.back_encoder,
        )

    def set_modules(self, modules):
        (
            self.dynamics,
            self.statenoise,
            self.likelihood,
            self.obs_encoder,
            self.back_encoder,
        ) = modules

    def save_model(self, file) -> None:
        with open(file, "wb") as f:
            spec = json.dumps(self.spec)
            f.write((spec + "\n").encode())
            eqx.tree_serialise_leaves(f, self.modules())

    @classmethod
    def load_model(cls, file) -> Self:
        with open(file, "rb") as f:
            spec = json.loads(f.readline().decode())
            model = XFADS(**spec)
            modules = eqx.tree_deserialise_leaves(f, model.modules())
            model.set_modules(modules)
            return model
