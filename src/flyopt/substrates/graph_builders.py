"""Graph transforms for the null-model family (PROTOCOL.md section 3).

Each function takes a reference weighted directed adjacency (scipy sparse,
weights[i, j] = signed synaptic weight from j onto i) and returns a
transformed adjacency that destroys one specific property while preserving
the rest, so that a difference between the real connectome and a null model
can be attributed to the property the null model destroyed.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def synthetic_reference_graph(
    n_units: int, density: float, seed: int, frac_excitatory: float = 0.8
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Build a synthetic signed weighted graph for testing the framework
    without real FlyWire data. Not used once real data is available.

    Returns (weights_csr, source_sign) where source_sign[j] in {+1, -1} is
    the excitatory/inhibitory identity of unit j (all of j's outgoing edges
    carry that sign, mirroring Dale's law as FlyWire data does).
    """
    rng = np.random.default_rng(seed)
    n_edges = int(round(density * n_units * n_units))
    rows = rng.integers(0, n_units, size=n_edges)
    cols = rng.integers(0, n_units, size=n_edges)
    source_sign = np.where(rng.random(n_units) < frac_excitatory, 1.0, -1.0)
    magnitudes = rng.lognormal(mean=0.0, sigma=0.5, size=n_edges)
    data = magnitudes * source_sign[cols]
    weights = sparse.csr_matrix((data, (rows, cols)), shape=(n_units, n_units))
    weights.sum_duplicates()
    return weights, source_sign


def er_null(reference: sparse.csr_matrix, seed: int) -> sparse.csr_matrix:
    """Erdos-Renyi null: same shape and edge count, edges placed uniformly
    at random (self-loops excluded), weights resampled i.i.d. from the
    reference's empirical weight distribution. Destroys degree distribution,
    reciprocity and every higher-order structure — the weakest null.

    Vectorized batched-rejection sampling: at FlyWire's density
    (~15M edges / 139k^2 pairs, collision rate low) this converges in a
    couple of oversampled rounds, unlike a Python-level rejection loop
    which does not scale to millions of edges.
    """
    rng = np.random.default_rng(seed)
    n = reference.shape[0]
    n_edges = reference.nnz
    ref_weights = reference.data

    keys = np.empty(0, dtype=np.int64)
    while keys.size < n_edges:
        need = n_edges - keys.size
        oversample = int(need * 1.15) + 1000
        r = rng.integers(0, n, size=oversample, dtype=np.int64)
        c = rng.integers(0, n, size=oversample, dtype=np.int64)
        mask = r != c
        batch_keys = r[mask] * n + c[mask]
        keys = np.unique(np.concatenate([keys, batch_keys]))
        if keys.size > n_edges:
            sel = rng.choice(keys.size, size=n_edges, replace=False)
            keys = keys[sel]

    rows = keys // n
    cols = keys % n
    data = rng.choice(ref_weights, size=n_edges, replace=True)
    out = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    return out


def degree_preserving_rewire(
    reference: sparse.csr_matrix, seed: int, n_sweeps: int = 8
) -> sparse.csr_matrix:
    """Maslov-Sneppen directed double-edge swap: preserves each unit's
    in-degree and out-degree exactly, destroys everything else (which
    neurons specifically connect to which). This is the primary null model
    the falsification criterion (PROTOCOL.md section 2) is measured against.

    Degree preservation holds *by construction*, not just empirically: src
    (presynaptic id) is never modified for any edge, and a pair's dst
    values are only ever exchanged as a unit — either both swap or neither
    does — so the multiset of src values (out-degree) and the multiset of
    dst values (in-degree) are exactly invariant, with no revert/backstop
    step needed to restore it after the fact.

    Vectorized batched implementation: each sweep randomly pairs up all
    edges once and proposes swapping dst between each pair. A pair is only
    ever *accepted* — verified, never applied-then-undone — once its
    resulting key (src, new_dst) is confirmed not to collide with a
    self-loop, with any other accepted pair's new key, or with any
    rejected pair's (unchanged) original key. Since the original graph's
    n_edges keys are already pairwise distinct, only accepted pairs can
    ever introduce a fresh collision, so iterating "demote any accepted
    pair involved in a collision, recheck" converges monotonically (the
    accepted set only shrinks) to a fully consistent, exactly-valid state.
    A sequential one-swap-at-a-time loop does not scale to FlyWire's edge
    count; this does (tens of seconds total).
    """
    rng = np.random.default_rng(seed)
    coo = reference.tocoo()
    src = coo.col.copy()  # edge j -> i stored as weights[i, j]
    dst = coo.row.copy()
    weights = coo.data.copy()
    n_edges = len(src)
    n = reference.shape[0]

    for _ in range(n_sweeps):
        perm = rng.permutation(n_edges)
        half = n_edges // 2
        e1 = perm[:half]
        e2 = perm[half : 2 * half]

        orig_key_e1 = src[e1].astype(np.int64) * n + dst[e1].astype(np.int64)
        orig_key_e2 = src[e2].astype(np.int64) * n + dst[e2].astype(np.int64)
        new_dst_e1 = dst[e2]
        new_dst_e2 = dst[e1]
        new_key_e1 = src[e1].astype(np.int64) * n + new_dst_e1.astype(np.int64)
        new_key_e2 = src[e2].astype(np.int64) * n + new_dst_e2.astype(np.int64)

        accepted = (src[e1] != new_dst_e1) & (src[e2] != new_dst_e2)  # no self-loop

        for _ in range(100):
            key_e1 = np.where(accepted, new_key_e1, orig_key_e1)
            key_e2 = np.where(accepted, new_key_e2, orig_key_e2)
            _, inverse, counts = np.unique(
                np.concatenate([key_e1, key_e2]), return_inverse=True, return_counts=True
            )
            dup = counts[inverse] > 1
            newly_bad = accepted & (dup[:half] | dup[half:])
            if not newly_bad.any():
                break
            accepted = accepted & ~newly_bad

        dst[e1[accepted]] = new_dst_e1[accepted]
        dst[e2[accepted]] = new_dst_e2[accepted]

    out = sparse.csr_matrix((weights, (dst, src)), shape=(n, n))
    return out


def weight_shuffle(reference: sparse.csr_matrix, seed: int) -> sparse.csr_matrix:
    """Same topology (identical edge set), weight magnitudes and signs
    permuted among existing edges. Destroys any relationship between where
    an edge is and how strong it is, keeps pure topology fixed.
    """
    rng = np.random.default_rng(seed)
    coo = reference.tocoo()
    shuffled = coo.data.copy()
    rng.shuffle(shuffled)
    return sparse.csr_matrix((shuffled, (coo.row, coo.col)), shape=reference.shape)


def scale_free_null(reference: sparse.csr_matrix, seed: int) -> sparse.csr_matrix:
    """2026-09-20: a DIFFERENT kind of null than the other four -- not
    derived from `reference`'s own degree sequence at all (unlike
    `degree_preserving_rewire`, which reuses the real fly's EXACT degree
    sequence). Instead, samples a FRESH power-law (heavy-tailed) degree
    sequence, independent of anything fly-specific, matched only to
    `reference`'s node count and total edge count. Answers a different
    question than the other nulls: does the fly connectome's measured
    advantage require its OWN specific hub structure, or does ANY graph
    from the general "heavy-tailed/hub-dominated" class do just as well?
    If this null also beats er_null-like uniform graphs, the advantage is
    a property of the topology CLASS (heavy-tailed degree), not of the
    fly's connectome identity -- see EXPERIMENTS.md 2026-09-20.

    Built via configuration-model stub-matching (independent out/in-degree
    sequences sampled from a Pareto distribution, rescaled to the same
    total edge count as `reference`), then self-loops and duplicate edges
    are dropped (a small, expected discrepancy from the target edge count,
    same convention as any configuration-model realization). Weight
    magnitudes+signs are resampled i.i.d. from `reference`'s own empirical
    weight distribution (matching `weight_shuffle`'s convention) so the
    ONLY thing that differs from `reference` is the topology's origin.
    """
    rng = np.random.default_rng(seed)
    n = reference.shape[0]
    coo = reference.tocoo()
    n_edges_target = coo.nnz
    weight_pool = coo.data

    def _pareto_degree_sequence(total: int, pareto_shape: float = 2.3) -> np.ndarray:
        raw = rng.pareto(pareto_shape, size=n) + 1.0
        seq = np.round(raw * (total / raw.sum())).astype(np.int64)
        seq = np.clip(seq, 0, None)
        diff = total - seq.sum()
        step = 1 if diff > 0 else -1
        idx = rng.choice(n, size=abs(int(diff)), replace=True)
        for i in idx:
            if step < 0 and seq[i] == 0:
                continue
            seq[i] += step
        return seq

    out_deg = _pareto_degree_sequence(n_edges_target)
    in_deg = _pareto_degree_sequence(n_edges_target)

    out_stubs = np.repeat(np.arange(n), out_deg)
    in_stubs = np.repeat(np.arange(n), in_deg)
    m = min(len(out_stubs), len(in_stubs))
    out_stubs = out_stubs[:m]
    in_stubs = in_stubs[:m]
    rng.shuffle(in_stubs)

    keys = out_stubs.astype(np.int64) * n + in_stubs.astype(np.int64)
    self_loop = out_stubs == in_stubs
    _, first_idx = np.unique(keys[~self_loop], return_index=True)
    src = out_stubs[~self_loop][first_idx]
    dst = in_stubs[~self_loop][first_idx]

    weights = rng.choice(weight_pool, size=len(src), replace=True)
    return sparse.csr_matrix((weights, (dst, src)), shape=(n, n))


def scale_free_correlated_null(reference: sparse.csr_matrix, seed: int) -> sparse.csr_matrix:
    """2026-09-20 direct follow-up to `scale_free_null`: that null sampled
    out-degree and in-degree INDEPENDENTLY and lost decisively to the real
    connectome (Cliff's delta=0.738, n=30). A structural analysis then
    found why -- in the real graph, a neuron's in-degree and out-degree
    are strongly correlated (Spearman r=0.70: high-input "integration"
    neurons tend to also be high-output "broadcast" neurons), and
    `degree_preserving_rewire` (which wins) automatically preserves this
    because it keeps each node's own (in-degree, out-degree) pair fixed.
    `scale_free_null`'s independent sampling destroyed it (r=-0.03).

    This null tests that mechanism directly: same power-law degree
    MAGNITUDE distribution as `scale_free_null`, but out-degree and
    in-degree are now derived from a SHARED per-node "hub intensity"
    value (plus independent noise), reproducing a similar in/out
    correlation to the real graph instead of sampling them independently.
    If this null now performs like `degree_preserving_rewire` (rather
    than like `scale_free_null`), the in/out-degree correlation itself is
    confirmed as (most of) the load-bearing feature -- see EXPERIMENTS.md
    2026-09-20 "Mekanizma netlesti".
    """
    rng = np.random.default_rng(seed)
    n = reference.shape[0]
    coo = reference.tocoo()
    n_edges_target = coo.nnz
    weight_pool = coo.data

    hub_intensity = rng.pareto(2.3, size=n) + 1.0
    out_noise = rng.pareto(4.0, size=n) + 1.0
    in_noise = rng.pareto(4.0, size=n) + 1.0
    out_raw = hub_intensity * out_noise
    in_raw = hub_intensity * in_noise

    def _to_sequence(raw: np.ndarray, total: int) -> np.ndarray:
        seq = np.round(raw * (total / raw.sum())).astype(np.int64)
        seq = np.clip(seq, 0, None)
        diff = total - seq.sum()
        step = 1 if diff > 0 else -1
        idx = rng.choice(n, size=abs(int(diff)), replace=True)
        for i in idx:
            if step < 0 and seq[i] == 0:
                continue
            seq[i] += step
        return seq

    out_deg = _to_sequence(out_raw, n_edges_target)
    in_deg = _to_sequence(in_raw, n_edges_target)

    out_stubs = np.repeat(np.arange(n), out_deg)
    in_stubs = np.repeat(np.arange(n), in_deg)
    m = min(len(out_stubs), len(in_stubs))
    out_stubs = out_stubs[:m]
    in_stubs = in_stubs[:m]
    rng.shuffle(in_stubs)

    keys = out_stubs.astype(np.int64) * n + in_stubs.astype(np.int64)
    self_loop = out_stubs == in_stubs
    _, first_idx = np.unique(keys[~self_loop], return_index=True)
    src = out_stubs[~self_loop][first_idx]
    dst = in_stubs[~self_loop][first_idx]

    weights = rng.choice(weight_pool, size=len(src), replace=True)
    return sparse.csr_matrix((weights, (dst, src)), shape=(n, n))


def community_preserving_rewire(
    reference: sparse.csr_matrix, communities: np.ndarray, seed: int, n_sweeps: int = 8
) -> sparse.csr_matrix:
    """2026-09-20 direct follow-up to the `scale_free_null`/`scale_free_correlated_null`
    falsifications: neither independent nor in/out-correlated synthetic
    degree sequences closed the gap to the real connectome. A structural
    analysis then found real has much higher modularity (Louvain Q=0.357,
    avg clustering=0.276) than EVERY null tried so far -- including
    `degree_preserving_rewire` itself (Q=0.069, clustering=0.103), which
    the plain Maslov-Sneppen double-edge-swap destroys along with
    everything else except the raw degree sequence.

    This preserves BOTH the exact degree sequence (same guarantee as
    `degree_preserving_rewire`) AND the real community-to-community edge
    count matrix (i.e. modularity/block structure) by only ever swapping
    the endpoint of two edges that share the same (community(src),
    community(dst)) bucket -- so an edge that went from community 3 to
    community 7 can only be re-targeted to another node that ALSO
    receives an edge from community 3, keeping every inter/intra-community
    edge count exactly fixed. Only WHICH specific node within the
    appropriate community receives a given src's edge is randomized.

    `communities`: an array of length n giving each node's community id
    (e.g. from `networkx.algorithms.community.louvain_communities` on the
    REAL reference graph -- computed once, reused for every seed/null so
    the block structure being preserved is always the real one).
    """
    rng = np.random.default_rng(seed)
    coo = reference.tocoo()
    src = coo.col.copy()
    dst = coo.row.copy()
    weights = coo.data.copy()
    n_edges = len(src)
    n = reference.shape[0]
    n_comm = int(communities.max()) + 1

    bucket_key = communities[src].astype(np.int64) * n_comm + communities[dst].astype(np.int64)
    order = np.argsort(bucket_key, kind="stable")
    sorted_keys = bucket_key[order]
    _, bucket_starts = np.unique(sorted_keys, return_index=True)
    bucket_starts = np.append(bucket_starts, n_edges)

    for _ in range(n_sweeps):
        for bi in range(len(bucket_starts) - 1):
            idxs = order[bucket_starts[bi]:bucket_starts[bi + 1]]
            n_bucket = len(idxs)
            if n_bucket < 2:
                continue
            perm = rng.permutation(n_bucket)
            half = n_bucket // 2
            e1 = idxs[perm[:half]]
            e2 = idxs[perm[half:2 * half]]

            orig_key_e1 = src[e1].astype(np.int64) * n + dst[e1].astype(np.int64)
            orig_key_e2 = src[e2].astype(np.int64) * n + dst[e2].astype(np.int64)
            new_dst_e1 = dst[e2]
            new_dst_e2 = dst[e1]
            new_key_e1 = src[e1].astype(np.int64) * n + new_dst_e1.astype(np.int64)
            new_key_e2 = src[e2].astype(np.int64) * n + new_dst_e2.astype(np.int64)

            accepted = (src[e1] != new_dst_e1) & (src[e2] != new_dst_e2)

            for _ in range(100):
                key_e1 = np.where(accepted, new_key_e1, orig_key_e1)
                key_e2 = np.where(accepted, new_key_e2, orig_key_e2)
                _, inverse, counts = np.unique(
                    np.concatenate([key_e1, key_e2]), return_inverse=True, return_counts=True
                )
                dup = counts[inverse] > 1
                newly_bad = accepted & (dup[:half] | dup[half:])
                if not newly_bad.any():
                    break
                accepted = accepted & ~newly_bad

            dst[e1[accepted]] = new_dst_e1[accepted]
            dst[e2[accepted]] = new_dst_e2[accepted]

    out = sparse.csr_matrix((weights, (dst, src)), shape=(n, n))
    return out


def sign_flip(
    reference: sparse.csr_matrix, source_sign: np.ndarray, seed: int
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Shuffle which units are excitatory vs inhibitory (E/I identity is
    reassigned across units, preserving the global E/I ratio and Dale's law
    — each unit's outgoing edges stay uniformly signed), while keeping
    topology and weight magnitudes fixed. Destroys any structure specific to
    *which* neurons are inhibitory.
    """
    rng = np.random.default_rng(seed)
    n = reference.shape[0]
    permuted_sign = source_sign.copy()
    rng.shuffle(permuted_sign)

    coo = reference.tocoo()
    magnitudes = np.abs(coo.data)
    new_data = magnitudes * permuted_sign[coo.col]
    out = sparse.csr_matrix((new_data, (coo.row, coo.col)), shape=reference.shape)
    return out, permuted_sign
