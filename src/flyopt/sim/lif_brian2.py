"""Brian2 reference LIF implementation, independent of lif_vectorized.py.

Purpose: an equivalence test (tests/test_lif_equivalence.py) that proves the
fast vectorized engine reproduces the same spike trains as a well-established
simulator, within numerical tolerance. This module is not meant to be fast —
it exists only to catch bugs in the vectorized engine, so it deliberately
mirrors the discrete-time Euler update in lif_vectorized.py rather than using
Brian2's continuous-time solvers (which would not admit a direct comparison).
"""
from __future__ import annotations

import numpy as np

from flyopt.sim.lif_vectorized import LIFParams


def simulate_discrete_euler_reference(
    weights_csr,
    stimulus_sequence: np.ndarray,
    seed: int,
    params: LIFParams = LIFParams(),
) -> np.ndarray:
    """Brian2-driven reference simulation using the *same* discrete Euler
    update, one dt per net.run() call, recurrent input computed from the
    previous step's spikes exactly as SparseLIFEngine.step does — so outputs
    are directly comparable.

    stimulus_sequence: shape (n_steps, n_units)
    returns: spike_record, shape (n_steps, n_units), dtype float32
    """
    import brian2 as b2

    b2.device.reinit()
    b2.device.activate()
    b2.defaultclock.dt = params.dt_ms * b2.ms

    n_units = weights_csr.shape[0]
    n_steps = stimulus_sequence.shape[0]

    rng = np.random.default_rng(seed)
    v0 = rng.random(n_units) * params.v_threshold

    eqs = """
    dv/dt = ((-(v - v_rest) + I_ext + I_rec) / tau_m) : 1 (unless refractory)
    I_ext : 1
    I_rec : 1
    v_rest : 1
    tau_m : second
    """
    G = b2.NeuronGroup(
        n_units,
        eqs,
        threshold=f"v >= {params.v_threshold}",
        reset=f"v = {params.v_reset}",
        refractory=params.refractory_ms * b2.ms,
        method="euler",
    )
    G.v = v0
    G.v_rest = params.v_rest
    G.tau_m = params.tau_m_ms * b2.ms

    spike_mon = b2.SpikeMonitor(G)
    net = b2.Network(G, spike_mon)

    spike_record = np.zeros((n_steps, n_units), dtype=np.float32)
    last_spikes = np.zeros(n_units, dtype=np.float32)
    dt_ms = params.dt_ms

    for t in range(n_steps):
        G.I_ext = stimulus_sequence[t]
        G.I_rec = weights_csr @ last_spikes
        n_before = spike_mon.num_spikes
        net.run(dt_ms * b2.ms)
        new_i = spike_mon.i[n_before:]
        spikes = np.zeros(n_units, dtype=np.float32)
        if len(new_i) > 0:
            spikes[np.asarray(new_i)] = 1.0
        spike_record[t] = spikes
        last_spikes = spikes

    return spike_record
