import numpy as np

from flyopt.substrate import Substrate
from flyopt.substrates.graph_builders import synthetic_reference_graph
from flyopt.substrates.random_mlp import RandomMLPSubstrate
from flyopt.substrates.sparse_recurrent import SparseRecurrentSubstrate


def test_sparse_recurrent_substrate_satisfies_protocol():
    ref, _ = synthetic_reference_graph(n_units=50, density=0.05, seed=1)
    sub = SparseRecurrentSubstrate(ref)
    assert isinstance(sub, Substrate)
    assert sub.n_units == 50


def test_random_mlp_substrate_satisfies_protocol():
    sub = RandomMLPSubstrate(n_units=50, n_params_target=500, seed_init=1)
    assert isinstance(sub, Substrate)
    assert sub.n_units == 50


def test_sparse_recurrent_step_reproducible_given_seed():
    ref, _ = synthetic_reference_graph(n_units=60, density=0.05, seed=2)
    sub = SparseRecurrentSubstrate(ref)
    stimulus = np.ones(60) * 0.5

    state_a = sub.reset(seed=123)
    state_b = sub.reset(seed=123)
    spikes_a, _ = sub.step(stimulus, state_a)
    spikes_b, _ = sub.step(stimulus, state_b)
    np.testing.assert_array_equal(spikes_a, spikes_b)


def test_sparse_recurrent_accepts_and_ignores_reward():
    ref, _ = synthetic_reference_graph(n_units=30, density=0.05, seed=3)
    sub = SparseRecurrentSubstrate(ref)
    state = sub.reset(seed=1)
    stimulus = np.zeros(30)
    # must not raise when a reward is passed, even though this substrate has no plasticity
    spikes, _ = sub.step(stimulus, state, reward=1.0)
    assert spikes.shape == (30,)


def test_random_mlp_step_reproducible_given_seed():
    sub = RandomMLPSubstrate(n_units=20, n_params_target=200, seed_init=7)
    stimulus = np.linspace(-1, 1, 20)

    state_a = sub.reset(seed=5)
    state_b = sub.reset(seed=5)
    out_a, _ = sub.step(stimulus, state_a)
    out_b, _ = sub.step(stimulus, state_b)
    np.testing.assert_array_equal(out_a, out_b)
