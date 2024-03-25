import pytest

from xfads.spec import ModelSpec


@pytest.fixture
def spec():
    spec: ModelSpec = dict(
        neural_dim=10,
        state_dim=2,
        input_dim=0,
        covariate_dim=0,
        approx="DiagMVN",
        mc_size=10,
        seed=0,
        dyn_spec={"module": "Nonlinear", "depth": 2, "width": 2, "state_noise": 10.0},
        lik_spec={"emission_noise": 10.0},
        enc_spec={"depth": 2, "width": 2},
        opt_spec={
            "mode": "em",
            "max_inner_iter": 10,
            "max_outer_iter": 10,
            "learning_rate": 1e-3,
            "clip_norm": 1.0,
            "batch_size": 1,
            "static": "sl",
        },
    )

    return spec
