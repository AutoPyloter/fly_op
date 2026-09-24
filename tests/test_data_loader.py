import numpy as np
import pandas as pd

from flyopt.data.loader import NT_COLUMNS, build_adjacency


def _synthetic_connections() -> pd.DataFrame:
    # Neuron 10 is a clear ACH (excitatory) neuron: two outgoing rows across
    # different neuropils, should be aggregated into one edge per post.
    # Neuron 20 is a clear GABA (inhibitory) neuron.
    # Neuron 30 has no outgoing rows at all -> sign should default to 0.
    rows = [
        {"pre_pt_root_id": 10, "post_pt_root_id": 20, "neuropil": "AL", "syn_count": 5,
         "gaba_avg": 0.0, "ach_avg": 0.9, "glut_avg": 0.0, "oct_avg": 0.0, "ser_avg": 0.0, "da_avg": 0.0},
        {"pre_pt_root_id": 10, "post_pt_root_id": 20, "neuropil": "MB", "syn_count": 3,
         "gaba_avg": 0.0, "ach_avg": 0.8, "glut_avg": 0.1, "oct_avg": 0.0, "ser_avg": 0.0, "da_avg": 0.0},
        {"pre_pt_root_id": 20, "post_pt_root_id": 30, "neuropil": "AL", "syn_count": 4,
         "gaba_avg": 0.95, "ach_avg": 0.0, "glut_avg": 0.0, "oct_avg": 0.0, "ser_avg": 0.0, "da_avg": 0.0},
    ]
    return pd.DataFrame(rows)


def test_build_adjacency_aggregates_across_neuropil():
    connections = _synthetic_connections()
    root_ids = np.array([10, 20, 30])
    weights, source_sign, id_to_index = build_adjacency(connections, root_ids)

    i10, i20, i30 = id_to_index[10], id_to_index[20], id_to_index[30]

    # neuron 10 -> 20 aggregated syn_count = 5 + 3 = 8, sign +1 (ACH dominant)
    assert weights[i20, i10] == 8.0
    # neuron 20 -> 30 syn_count = 4, sign -1 (GABA dominant)
    assert weights[i30, i20] == -4.0


def test_build_adjacency_sign_per_neuron():
    connections = _synthetic_connections()
    root_ids = np.array([10, 20, 30])
    _, source_sign, id_to_index = build_adjacency(connections, root_ids)

    assert source_sign[id_to_index[10]] == 1.0
    assert source_sign[id_to_index[20]] == -1.0
    # neuron 30 has no outgoing edges -> undefined sign, not guessed
    assert source_sign[id_to_index[30]] == 0.0


def test_build_adjacency_shape_matches_root_ids_not_just_connected():
    connections = _synthetic_connections()
    root_ids = np.array([10, 20, 30, 40, 50])  # 40, 50 appear in no connection at all
    weights, source_sign, id_to_index = build_adjacency(connections, root_ids)
    assert weights.shape == (5, 5)
    assert len(source_sign) == 5
