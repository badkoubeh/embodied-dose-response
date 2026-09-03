"""Frozen-model wrapper (PLAN.md 3.2).

STUB -- needs weights and a >=24 GB GPU.

Loads weights once per job, then runs inference with a pinned temperature and a
fixed flow-matching seed.

`n_traj_samples` is the open question. PLAN.md 3.2 pins the runner to 1, but
PLAN.md 3.2 metrics ask for minADE-6 and PLAN.md 4 calibrates against NVIDIA's
published minADE-6 of 1.22 m -- which one sample cannot produce. If the answer is
6, sample the flow-matching decoder 6 times behind a SHARED VLM prefix: the grid
is 14.4k-30k inferences (12-33 GPU-h at 3-4 s), so naive end-to-end 6x sampling
is 72-200 GPU-h and breaks the ~100 GPU-h cap on its own. See the open issue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edr.schema import Sample


@dataclass(frozen=True, slots=True)
class Inference:
    """One forward pass: (K, 64, 3) trajectories, reasoning text, and cost."""

    trajectory: np.ndarray
    reasoning_text: str
    latency_s: float
    peak_vram_gb: float


class FrozenPolicy:
    """A loaded, frozen driving VLA."""

    def __init__(
        self,
        model_id: str,
        n_traj_samples: int = 1,
        temperature: float = 0.0,
        flow_seed: int = 0,
        device: str = "cuda",
    ) -> None:
        raise NotImplementedError("blocked on model weights (PLAN.md 5, Stage 0)")

    @property
    def weights_hash(self) -> str:
        """Identity of the loaded weights.

        Use the HF revision SHA plus the safetensors index digest. Do NOT digest
        the 22.2 GB of weights themselves -- that runs once per shard job, and
        there are many shard jobs.
        """
        raise NotImplementedError("blocked on model weights (PLAN.md 5, Stage 0)")

    def infer(self, sample: Sample, video: object) -> Inference:
        """Run one inference. Must be free of side effects across calls."""
        raise NotImplementedError("blocked on model weights (PLAN.md 5, Stage 0)")
