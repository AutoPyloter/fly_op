import json

import numpy as np

from flyopt.experiment.stats import (
    cliffs_delta,
    holm_bonferroni,
    mann_whitney,
    passes_falsification_gate,
)


def test_cliffs_delta_identical_distributions_near_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    assert abs(cliffs_delta(a, b)) < 0.15


def test_cliffs_delta_fully_separated_is_one():
    a = np.arange(10, 20)
    b = np.arange(0, 10)
    assert cliffs_delta(a, b) == 1.0
    assert cliffs_delta(b, a) == -1.0


def test_mann_whitney_detects_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(loc=2.0, size=40)
    b = rng.normal(loc=0.0, size=40)
    result = mann_whitney(a, b)
    assert result.p_value < 0.05
    assert result.effect_size > 0.33


def test_gate_requires_both_significance_and_effect_size():
    rng = np.random.default_rng(2)
    a = rng.normal(loc=0.05, size=200)
    b = rng.normal(loc=0.0, size=200)
    result = mann_whitney(a, b)
    # small shift, large n: can be "significant" with negligible effect size
    if result.p_value < 0.05 and abs(result.effect_size) <= 0.33:
        assert not passes_falsification_gate(result)


def test_holm_bonferroni_step_down():
    p_values = [0.001, 0.02, 0.03, 0.5]
    reject = holm_bonferroni(p_values, alpha=0.05)
    assert reject[0] is True
    assert reject[3] is False


def test_cliffs_delta_returns_native_float():
    # regression: numpy.float64 here silently produced a non-JSON-serializable
    # numpy.bool_ downstream in passes_falsification_gate whenever the p<0.05
    # branch wasn't short-circuited away (Faz 1b run crashed on this).
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert type(cliffs_delta(a, b)) is float


def test_passes_falsification_gate_result_is_json_serializable():
    rng = np.random.default_rng(3)
    a = rng.normal(loc=2.0, size=40)
    b = rng.normal(loc=0.0, size=40)
    result = mann_whitney(a, b)
    assert result.p_value < 0.05  # exercises the non-short-circuited branch
    gate_pass = passes_falsification_gate(result)
    json.dumps({"passes_falsification_gate": gate_pass})  # must not raise
