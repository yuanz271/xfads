from tempfile import TemporaryDirectory
from jax import numpy as jnp, random as jrandom
import chex
import equinox as eqx

from xfads.smoother import XFADS
from xfads.spec import ModelSpec


def test_xfads(spec: ModelSpec, capsys):
    N: int = 100
    T: int = 100

    xfads = XFADS(**spec)
    
    t = jnp.tile(jnp.arange(T, dtype=int), (N, 1))
    y = jrandom.normal(key=jrandom.key(0), shape=(N, T, spec['observation_dim']))
    u = jrandom.normal(key=jrandom.key(0), shape=(N, T, spec['input_dim']))
    x = jnp.zeros((N, T, spec['covariate_dim']))

    with capsys.disabled():
        xfads.fit((t, y, u), key=jrandom.key(0), mode="joint")

    with TemporaryDirectory() as tmpdir:
        xfads.save_model(f"{tmpdir}/model.eqx")
        loaded_model = XFADS.load_model(f"{tmpdir}/model.eqx")

    chex.assert_trees_all_equal(
        eqx.filter(xfads.modules()[0], eqx.is_array),
        eqx.filter(loaded_model.modules()[0], eqx.is_array),
    )
