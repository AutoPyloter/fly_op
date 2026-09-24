import numpy as np

from flyopt.benchmarks import SPHERE_BOUNDS, sphere
from flyopt.experiment.search_loop import run_hillclimb
from flyopt.substrates.graph_builders import synthetic_reference_graph
from flyopt.substrates.sparse_recurrent import SparseRecurrentSubstrate
from flyopt.variants.fly_proposer import FlyProposer, FlyProposerConfig


def test_hillclimb_is_monotonically_non_increasing_in_best_fx():
    ref, _ = synthetic_reference_graph(n_units=200, density=0.05, seed=1)
    substrate = SparseRecurrentSubstrate(ref)
    config = FlyProposerConfig(dim=5, n_readout=20, T=5, step_scale=1.0, decode_scale=0.3)
    proposer = FlyProposer(substrate, config, seed=42)

    rng = np.random.default_rng(7)
    initial_x = rng.uniform(*SPHERE_BOUNDS, size=5)
    initial_fx = sphere(initial_x)

    best_fx = run_hillclimb(proposer, sphere, dim=5, bounds=SPHERE_BOUNDS, budget=200, seed=7)
    assert best_fx <= initial_fx


def test_fly_proposer_reproducible_given_seed():
    ref, _ = synthetic_reference_graph(n_units=150, density=0.05, seed=2)
    config = FlyProposerConfig(dim=4, n_readout=15, T=4)

    pa = FlyProposer(SparseRecurrentSubstrate(ref), config, seed=3)
    pb = FlyProposer(SparseRecurrentSubstrate(ref), config, seed=3)
    fa = run_hillclimb(pa, sphere, dim=4, bounds=SPHERE_BOUNDS, budget=50, seed=9)
    fb = run_hillclimb(pb, sphere, dim=4, bounds=SPHERE_BOUNDS, budget=50, seed=9)
    assert fa == fb


def test_same_seed_gives_same_encode_decode_across_different_substrates():
    # Different topologies (same n_units): encode/decode indices and the
    # readout matrix must depend only on seed, never on the substrate, so
    # the proposal mechanism is identical across flywire vs null runs.
    ref_a, _ = synthetic_reference_graph(n_units=100, density=0.05, seed=5)
    ref_b, _ = synthetic_reference_graph(n_units=100, density=0.05, seed=6)
    config = FlyProposerConfig(dim=3, n_readout=10, T=3)

    pa = FlyProposer(SparseRecurrentSubstrate(ref_a), config, seed=11)
    pb = FlyProposer(SparseRecurrentSubstrate(ref_b), config, seed=11)

    np.testing.assert_array_equal(pa.encode_indices, pb.encode_indices)
    np.testing.assert_array_equal(pa.decode_indices, pb.decode_indices)
    np.testing.assert_array_equal(pa.readout_W, pb.readout_W)


def test_encode_and_decode_indices_disjoint():
    ref, _ = synthetic_reference_graph(n_units=80, density=0.05, seed=8)
    config = FlyProposerConfig(dim=5, n_readout=10, T=2)
    p = FlyProposer(SparseRecurrentSubstrate(ref), config, seed=1)
    assert len(set(p.encode_indices.tolist()) & set(p.decode_indices.tolist())) == 0


def test_explicit_encode_decode_pools_are_respected():
    ref, _ = synthetic_reference_graph(n_units=100, density=0.05, seed=9)
    config = FlyProposerConfig(dim=4, n_readout=6, T=2)
    encode_pool = np.arange(0, 20)
    decode_pool = np.arange(50, 100)
    p = FlyProposer(SparseRecurrentSubstrate(ref), config, seed=2, encode_pool=encode_pool, decode_pool=decode_pool)
    assert set(p.encode_indices.tolist()) <= set(encode_pool.tolist())
    assert set(p.decode_indices.tolist()) <= set(decode_pool.tolist())


def test_readout_lr_zero_leaves_readout_unchanged():
    ref, _ = synthetic_reference_graph(n_units=80, density=0.05, seed=20)
    config = FlyProposerConfig(dim=4, n_readout=10, T=3, readout_lr=0.0)
    p = FlyProposer(SparseRecurrentSubstrate(ref), config, seed=1)
    W_before = p.readout_W.copy()
    x = np.zeros(4)
    ctx = None
    from flyopt.substrate import SearchContext
    ctx = SearchContext(iteration=0, budget_total=10, budget_used=0, best_x=x, best_fx=0.0)
    candidate = p.propose(x, ctx)
    p.tell(candidate, 0.0, improved=True)
    np.testing.assert_array_equal(p.readout_W, W_before)


def test_readout_lr_positive_updates_readout_on_improvement():
    ref, _ = synthetic_reference_graph(n_units=80, density=0.05, seed=21)
    config = FlyProposerConfig(dim=4, n_readout=10, T=3, readout_lr=0.1)
    p = FlyProposer(SparseRecurrentSubstrate(ref), config, seed=1)
    W_before = p.readout_W.copy()
    from flyopt.substrate import SearchContext
    x = np.zeros(4)
    ctx = SearchContext(iteration=0, budget_total=10, budget_used=0, best_x=x, best_fx=0.0)
    candidate = p.propose(x, ctx)
    p.tell(candidate, 0.0, improved=True)
    assert not np.array_equal(p.readout_W, W_before)


def test_readout_lr_stays_no_op_before_first_propose_call():
    ref, _ = synthetic_reference_graph(n_units=60, density=0.05, seed=22)
    config = FlyProposerConfig(dim=3, n_readout=8, T=2, readout_lr=0.1)
    p = FlyProposer(SparseRecurrentSubstrate(ref), config, seed=1)
    W_before = p.readout_W.copy()
    p.tell(np.zeros(3), 0.0, improved=True)  # tell() before any propose()
    np.testing.assert_array_equal(p.readout_W, W_before)


def test_overlapping_pools_raise():
    ref, _ = synthetic_reference_graph(n_units=50, density=0.05, seed=10)
    config = FlyProposerConfig(dim=3, n_readout=3, T=2)
    encode_pool = np.arange(0, 10)
    decode_pool = np.arange(5, 15)  # overlaps with encode_pool
    try:
        FlyProposer(SparseRecurrentSubstrate(ref), config, seed=3, encode_pool=encode_pool, decode_pool=decode_pool)
        assert False, "expected AssertionError for overlapping pools"
    except AssertionError:
        pass
