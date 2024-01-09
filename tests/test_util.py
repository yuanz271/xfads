from jax import random as jrnd
from bfs.util import Context


def test_context(dimensions):
    state_dim, input_dim, observation_dim = dimensions
    ctx = Context(jrnd.PRNGKey(0), state_dim, input_dim, observation_dim, 10)
    ctx.newkey()
    