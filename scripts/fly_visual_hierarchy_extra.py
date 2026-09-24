"""Extra 8 seeds (8-15) for visual_hierarchy only -- the strongest signal
(delta=0.438, underpowered at n=8) among the anatomical circuits.
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")
from fly_anatomical_circuits_multiseed import run_circuit, SEEDS as _  # noqa: E402
import fly_anatomical_circuits_multiseed as m

m.SEEDS = list(range(8, 16))


def main():
    p, delta = run_circuit("visual_hierarchy_extra", "lamina", "lobula_plate")
    print(f"\nvisual_hierarchy (seeds 8-15): p={p:.4f} delta={delta:+.3f}")


if __name__ == "__main__":
    main()
