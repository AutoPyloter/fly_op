import numpy as np
import pytest

from flyopt.sim.lif_vectorized import LIFParams, SparseLIFEngine
from flyopt.substrates.graph_builders import synthetic_reference_graph

brian2 = pytest.importorskip("brian2")


def test_vectorized_matches_brian2_reference_small_network():
    from flyopt.sim.lif_brian2 import simulate_discrete_euler_reference

    n_units = 8
    n_steps = 30
    params = LIFParams(tau_m_ms=20.0, v_threshold=1.0, refractory_ms=2.0, dt_ms=1.0)

    weights, _ = synthetic_reference_graph(n_units=n_units, density=0.15, seed=42)
    rng = np.random.default_rng(0)
    stimulus_sequence = rng.uniform(0.0, 0.3, size=(n_steps, n_units)).astype(np.float32)

    engine = SparseLIFEngine(weights, params)
    state = engine.reset(seed=1)
    vec_spikes = np.zeros((n_steps, n_units), dtype=np.float32)
    for t in range(n_steps):
        spikes, state = engine.step(stimulus_sequence[t], state)
        vec_spikes[t] = spikes

    ref_spikes = simulate_discrete_euler_reference(weights, stimulus_sequence, seed=1, params=params)

    # exact spike-for-spike match is the bar: both engines implement the same
    # discrete Euler update from the same initial v and the same input.
    np.testing.assert_array_equal(vec_spikes, ref_spikes)
