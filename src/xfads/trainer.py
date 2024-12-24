from dataclasses import dataclass

import jax
from jax import numpy as jnp, random as jrandom
from jax._src.random import KeyArray
import optax
import equinox as eqx
from tqdm import trange


@dataclass
class Opt:
    min_iter: int = 0
    max_inner_iter: int = 1
    max_em_iter: int = 1
    learning_rate: float = 1e-3
    clip_norm: float = 5.0
    batch_size: int = 1
    weight_decay: float = 1e-3
    

def make_optimizer(modules, opt: Opt):
    optimizer = optax.chain(
        optax.clip_by_global_norm(opt.clip_norm),
        optax.adamw(opt.learning_rate, weight_decay=opt.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(modules, eqx.is_inexact_array))

    return optimizer, opt_state


def train_batches(params, loss_value_and_grad, minibatches, optimizer, opt_state):
    
    @eqx.filter_jit
    def step(carry, batch):
        params, opt_state, acc_loss, n = carry
        loss, grads = loss_value_and_grad(params, *batch)
        acc_loss = acc_loss + loss  # acculumate losses
        n = n + 1
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return (params, opt_state, acc_loss, n), None
    
    acc_loss = 0.
    (params, opt_state, acc_loss, n), *_ = jax.lax.scan(step, init=(params, opt_state, acc_loss, 0), xs=minibatches)  
    # xs has to be tuple of arrays. scan iterates each along the first dimension
    # https://github.com/patrick-kidger/equinox/issues/558
    # make step a lambda

    total_loss = acc_loss / n
    
    return params, opt_state, total_loss


def minibatcher(*arrays, batch_size, key):
    # assert the arrays have the same leading dimension
    n_samples = len(arrays[0])
    n_batches = n_samples // batch_size
    indices = jrandom.permutation(key, n_samples)
    trimmed_sample_size = n_batches * batch_size  # clip remaining samples that are not enough for a batch
    trimmed_indices = indices[:trimmed_sample_size]

    keys = jrandom.split(key, n_batches)
    
    return (keys,) + tuple(jnp.reshape(arr[trimmed_indices], (n_batches, batch_size) + arr.shape[1:]) for arr in arrays)


def train_loop(modules, loss_value_and_grad, *data, key, opt: Opt):
    optimizer, opt_state = make_optimizer(modules, opt)
    
    batch_size = opt.batch_size    
    max_iter = opt.max_inner_iter
    
    def epoch(i, val):
        modules, opt_state, loss= val
        ekey = jrandom.fold_in(key, i)
        batches = minibatcher(*data, batch_size=batch_size, key=ekey)
        modules, opt_state, loss = train_batches(modules, loss_value_and_grad, batches, optimizer, opt_state)
        return modules, opt_state, loss
    
    loss = jnp.nan
    with trange(max_iter) as pbar:
        for i in pbar:
            modules, opt_state, loss, key = epoch(i, (modules, opt_state, loss))
            pbar.set_postfix({'loss': f"{loss:.3f}"})

    return modules, loss


# def train_loop(modules, loss_value_and_grad, *data, key, opt: Opt):
#     optimizer, opt_state = make_optimizer(modules, opt)
    
#     batch_size = opt.batch_size    
#     max_iter = opt.max_inner_iter
    
#     def epoch(i, val):
#         modules, opt_state, loss= val
#         ekey = jrandom.fold_in(key, i)
#         batches = minibatcher(*data, batch_size=batch_size, key=ekey)
#         modules, opt_state, loss = train_batches(modules, loss_value_and_grad, batches, optimizer, opt_state)
#         return modules, opt_state, loss
    
#     modules, opt_state, loss = jax.lax.fori_loop(0, max_iter, epoch, (modules, opt_state, jnp.nan))

#     return modules, loss

