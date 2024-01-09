import pytest


@pytest.fixture
def dimensions():
    state_dim = 2
    input_dim = 2
    observation_dim = 3
    return state_dim, input_dim, observation_dim
