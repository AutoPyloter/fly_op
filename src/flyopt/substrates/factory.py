"""Named substrate construction (PROTOCOL.md section 3).

Enforces the "never run only the real connectome" rule: `build_all` always
returns the real substrate alongside its nulls, and there is deliberately no
code path that returns a single substrate for an experiment run — see
experiment/runner.py, which takes a list, not one Substrate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from flyopt.substrates.graph_builders import (
    degree_preserving_rewire,
    er_null,
    sign_flip,
    weight_shuffle,
)
from flyopt.substrates.random_mlp import RandomMLPSubstrate
from flyopt.substrates.sparse_recurrent import SparseRecurrentSubstrate

DEFAULT_NULL_KINDS = ("degree_preserving", "weight_shuffle")
ALL_NULL_KINDS = ("er", "degree_preserving", "weight_shuffle", "sign_flip")


@dataclass
class NamedSubstrate:
    name: str
    substrate: SparseRecurrentSubstrate | RandomMLPSubstrate


def build_all(
    reference_weights: sparse.csr_matrix,
    source_sign: np.ndarray,
    seed: int,
    null_kinds: tuple[str, ...] = DEFAULT_NULL_KINDS,
    include_random_mlp: bool = False,
) -> list[NamedSubstrate]:
    """Build the real connectome substrate plus every requested null,
    all from the same seed so graph construction is reproducible per-run.
    """
    out = [NamedSubstrate("flywire", SparseRecurrentSubstrate(reference_weights))]

    if "er" in null_kinds:
        out.append(
            NamedSubstrate("er_null", SparseRecurrentSubstrate(er_null(reference_weights, seed)))
        )
    if "degree_preserving" in null_kinds:
        out.append(
            NamedSubstrate(
                "degree_preserving_null",
                SparseRecurrentSubstrate(degree_preserving_rewire(reference_weights, seed)),
            )
        )
    if "weight_shuffle" in null_kinds:
        out.append(
            NamedSubstrate(
                "weight_shuffle_null",
                SparseRecurrentSubstrate(weight_shuffle(reference_weights, seed)),
            )
        )
    if "sign_flip" in null_kinds:
        flipped, _ = sign_flip(reference_weights, source_sign, seed)
        out.append(NamedSubstrate("sign_flip_null", SparseRecurrentSubstrate(flipped)))

    if include_random_mlp:
        out.append(
            NamedSubstrate(
                "random_mlp",
                RandomMLPSubstrate(reference_weights.shape[0], reference_weights.nnz, seed),
            )
        )

    return out
