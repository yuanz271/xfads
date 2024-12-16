import jax
from jax import numpy as jnp, random as jrandom
import equinox as eqx
from xfads.nn import AddBias, VariantBiasLinear


def test_AddBias():
    key = jrandom.key(0)
    addb = AddBias(4, key=key)
    x = jnp.ones((4,))
    addb(x)


def test_VariantBiasLinear():
    key = jrandom.key(0)
    vblin = eqx.filter_jit(VariantBiasLinear(2, 4, 5, key=key))
    x = jnp.ones((2,))
    y = vblin(0, x)
