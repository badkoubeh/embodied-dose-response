#!/usr/bin/env python3
"""Submit the sharded sweep to SageMaker (PLAN.md 3.1).

STUB -- needs dataset access, weights, and a SageMaker role.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    raise NotImplementedError("blocked on dataset + weights access (PLAN.md 5, Stage 0)")


if __name__ == "__main__":
    main()
