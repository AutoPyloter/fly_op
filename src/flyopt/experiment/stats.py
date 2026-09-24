"""Statistics required by PROTOCOL.md section 4. Mean-only comparisons are
not implemented here on purpose — every comparison function returns an
effect size and a test, never a bare mean difference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats


@dataclass
class ComparisonResult:
    test: str
    statistic: float
    p_value: float
    effect_size_name: str
    effect_size: float
    n_a: int
    n_b: int


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta in [-1, 1]. Equivalent to 2*AUC - 1 for the
    Mann-Whitney U statistic; computed directly (O(n*m), fine at n=30).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    gt = sum((ai > b).sum() for ai in a)
    lt = sum((ai < b).sum() for ai in a)
    # cast to native float: numpy.float64 here would silently produce a
    # numpy.bool_ (not JSON-serializable) wherever this feeds a comparison
    # that isn't short-circuited away by an earlier native-bool False —
    # exactly what happened when Faz 1b's gate p-value was significant
    # (Faz 1's was not, so the buggy path was never exercised there).
    return float((gt - lt) / (len(a) * len(b)))


def mann_whitney(a: np.ndarray, b: np.ndarray) -> ComparisonResult:
    """Unpaired comparison (independent seeds between groups)."""
    stat, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
    return ComparisonResult(
        test="mann_whitney_u",
        statistic=float(stat),
        p_value=float(p),
        effect_size_name="cliffs_delta",
        effect_size=cliffs_delta(np.asarray(a), np.asarray(b)),
        n_a=len(a),
        n_b=len(b),
    )


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> ComparisonResult:
    """Paired comparison for shared-seed designs (PROTOCOL.md 4.3)."""
    a = np.asarray(a)
    b = np.asarray(b)
    stat, p = sp_stats.wilcoxon(a, b)
    return ComparisonResult(
        test="wilcoxon_signed_rank",
        statistic=float(stat),
        p_value=float(p),
        effect_size_name="cliffs_delta",
        effect_size=cliffs_delta(a, b),
        n_a=len(a),
        n_b=len(b),
    )


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Returns, in the original order of p_values, whether each is
    significant under the Holm-Bonferroni step-down procedure.
    """
    order = np.argsort(p_values)
    m = len(p_values)
    reject = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            reject[idx] = True
        else:
            break  # Holm's procedure stops at the first non-rejection
    return reject


GATE_EFFECT_SIZE_THRESHOLD = 0.33  # |Cliff's delta|, "orta etki" (PROTOCOL.md section 2)
GATE_ALPHA = 0.05


def passes_falsification_gate(result: ComparisonResult) -> bool:
    return result.p_value < GATE_ALPHA and abs(result.effect_size) > GATE_EFFECT_SIZE_THRESHOLD
