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

    y = jrandom.normal(key=jrandom.key(0), shape=(N, T, spec['neural_dim']))
    u = jrandom.normal(key=jrandom.key(0), shape=(N, T, spec['input_dim']))
    x = jnp.zeros((N, T, spec['covariate_dim']))

    with capsys.disabled():
        xfads.fit(y, u, x)

    with TemporaryDirectory() as tmpdir:
        xfads.save_model(f"{tmpdir}/model.eqx")
        loaded_model = XFADS.load_model(f"{tmpdir}/model.eqx")

    chex.assert_trees_all_equal(
        eqx.filter(xfads.modules["dynamics"], eqx.is_array),
        eqx.filter(loaded_model.modules["dynamics"], eqx.is_array),
    )
