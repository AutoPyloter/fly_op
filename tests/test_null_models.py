import numpy as np

from flyopt.substrates.graph_builders import (
    degree_preserving_rewire,
    er_null,
    sign_flip,
    synthetic_reference_graph,
    weight_shuffle,
)


def _in_out_degrees(weights_csr):
    out_deg = np.asarray((weights_csr != 0).sum(axis=0)).flatten()  # column j = outgoing from j
    in_deg = np.asarray((weights_csr != 0).sum(axis=1)).flatten()  # row i = incoming to i
    return in_deg, out_deg


def test_er_null_preserves_density_not_degrees():
    ref, _ = synthetic_reference_graph(n_units=200, density=0.02, seed=1)
    null = er_null(ref, seed=2)

    assert null.nnz == ref.nnz
    assert null.shape == ref.shape

    ref_in, ref_out = _in_out_degrees(ref)
    null_in, null_out = _in_out_degrees(null)
    # ER null should NOT preserve the exact degree sequence
    assert not np.array_equal(np.sort(ref_in), np.sort(null_in)) or not np.array_equal(
        np.sort(ref_out), np.sort(null_out)
    )


def test_degree_preserving_rewire_preserves_degree_sequence():
    ref, _ = synthetic_reference_graph(n_units=150, density=0.03, seed=3)
    rewired = degree_preserving_rewire(ref, seed=4, n_sweeps=8)

    assert rewired.nnz == ref.nnz

    ref_in, ref_out = _in_out_degrees(ref)
    rew_in, rew_out = _in_out_degrees(rewired)
    np.testing.assert_array_equal(np.sort(ref_in), np.sort(rew_in))
    np.testing.assert_array_equal(np.sort(ref_out), np.sort(rew_out))

    # topology should actually have changed (not a no-op)
    ref_edges = set(zip(*ref.nonzero()))
    rew_edges = set(zip(*rewired.nonzero()))
    assert ref_edges != rew_edges


def test_weight_shuffle_preserves_topology_and_weight_multiset():
    ref, _ = synthetic_reference_graph(n_units=120, density=0.03, seed=5)
    shuffled = weight_shuffle(ref, seed=6)

    ref_edges = set(zip(*ref.nonzero()))
    shuf_edges = set(zip(*shuffled.nonzero()))
    assert ref_edges == shuf_edges

    np.testing.assert_allclose(
        np.sort(np.abs(ref.data)), np.sort(np.abs(shuffled.data)), rtol=1e-6
    )


def test_sign_flip_preserves_ei_ratio_and_topology():
    ref, source_sign = synthetic_reference_graph(n_units=100, density=0.03, seed=7)
    flipped, new_sign = sign_flip(ref, source_sign, seed=8)

    assert (new_sign > 0).sum() == (source_sign > 0).sum()
    ref_edges = set(zip(*ref.nonzero()))
    flip_edges = set(zip(*flipped.nonzero()))
    assert ref_edges == flip_edges
    np.testing.assert_allclose(
        np.sort(np.abs(ref.data)), np.sort(np.abs(flipped.data)), rtol=1e-6
    )


def test_null_models_are_seed_reproducible():
    ref, sign = synthetic_reference_graph(n_units=80, density=0.04, seed=9)
    a = degree_preserving_rewire(ref, seed=42)
    b = degree_preserving_rewire(ref, seed=42)
    assert set(zip(*a.nonzero())) == set(zip(*b.nonzero()))
