import jax
from jax import numpy as jnp, random as jrandom
import optax


def test_train_batches(capsys):
    from xfads.trainer import train_batches, minibatcher
    
    x = jrandom.normal(jrandom.key(0), (100, 5))
    W = jrandom.normal(jrandom.key(0), (5, 5))
    y = x @ W + jrandom.normal(jrandom.key(1), (100, 5)) * 0.1

    params = jrandom.normal(jrandom.key(0), (5, 5))
    
    def loss(w, key, x, y):
        return jnp.mean(jnp.square(x@w - y))
    
    key = jrandom.key(0)

    minibatches = minibatcher(x, y, batch_size=10, key=key)

    optimizer = optax.adam(learning_rate=0.01)
    opt_state= optimizer.init(params)

    with capsys.disabled(): 
        optimal_params, opt_state, opt_loss = train_batches(params, jax.value_and_grad(loss), minibatches, optimizer, opt_state)
        print(f"{opt_loss=}")
