#!/usr/bin/env python3
"""Reproduce a single cell in isolation (PLAN.md 2.4).

STUB for the inference half; the perturbation half already works.

This script is the executable statement of the determinism guarantee: given only
(scenario_id, axis, level_index, seed_index), regenerate that cell's perturbed
input bit-identically, on any machine, in any shard order, at any later date.
`edr.seeding` is what makes that hold.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    raise NotImplementedError("inference half blocked on weights (PLAN.md 5, Stage 0)")


if __name__ == "__main__":
    main()
