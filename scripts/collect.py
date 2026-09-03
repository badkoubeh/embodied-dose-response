#!/usr/bin/env python3
"""Collect shard artifacts, score them, and emit the scorecard (PLAN.md 3.1).

STUB.

Runs entirely on CPU against stored raw artifacts, and is re-runnable at no GPU
cost -- that is the whole payoff of the PLAN.md 2.1 plane split. Re-running this
after recalibrating the meta-action extractor must never require re-inference.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
