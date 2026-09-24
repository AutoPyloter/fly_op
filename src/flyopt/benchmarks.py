"""Faz 1 gate benchmark functions (PROTOCOL.md Faz 1). Minimization,
global optimum at x=0 with f(0)=0 for both.
"""
from __future__ import annotations

import numpy as np


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


SPHERE_BOUNDS = (-5.0, 5.0)


def rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


RASTRIGIN_BOUNDS = (-5.12, 5.12)
