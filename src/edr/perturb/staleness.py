"""Stale ego-state (PLAN.md 3.2, `staleness`).

Models a lagging localization estimate: the pose the policy is told is current is
actually the pose from tau milliseconds ago, while perception sees the present.
That is a real and common failure -- a localization stack running behind its
sensor feed -- and it is a pure state-input degradation.

**Scope: ego-state only, video untouched.** PLAN.md 3.2 describes feeding a whole
window tau old and notes that it "touches the video path, so it defeats the pixel
cache". Staling the video too would make this end-to-end sensor latency rather
than ego-state degradation, which sits outside the question PLAN.md 1 poses and
outside the state-input framing of the cross-paradigm claim in PLAN.md 7.
Restricting it to the ego channel keeps the axis on-thesis and preserves the
PLAN.md 2.5 pixel cache, so all three axes reuse one decode. See the open issue.

**This axis is deterministic.** Given tau, there is no draw to make, so a second
perturbation seed produces a bit-identical duplicate. `configs/axis/staleness.yaml`
sets `seeds: 1` accordingly; running two would burn GPU-hours on a copy.
"""

from __future__ import annotations

import numpy as np

from edr.perturb.base import PerturbationResult, register
from edr.schema import EGO_X, EGO_Y, Sample


@register
class Staleness:
    """Shift the ego-state history back by tau, holding the video at the present.

    Severity is tau in milliseconds. PLAN.md 3.2 specifies tau in multiples of the
    frame period; values off the grid are rounded to the nearest waypoint and the
    realized lag is logged.

    Where the shift runs off the start of the available history, the vacated
    waypoints are extrapolated backwards at constant velocity. The alternative --
    repeating the oldest pose -- would inject an artificial stationary segment,
    which is a different and much cruder perturbation than a lag.
    """

    name = "staleness"
    unit = "ms"

    def apply(
        self, sample: Sample, severity: float, rng: np.random.Generator
    ) -> PerturbationResult:
        # `rng` is unused: see the module docstring. It stays in the signature to
        # satisfy the Perturbation protocol.
        del rng

        if severity < 0:
            raise ValueError(f"severity must be non-negative, got {severity}")

        ego = sample.ego_history
        n = ego.shape[0]
        if n < 2:
            raise ValueError("staleness needs at least 2 waypoints to extrapolate")

        period_ms = sample.waypoint_dt_s * 1000.0
        shift = int(round(severity / period_ms))
        if shift == 0:
            return PerturbationResult(sample, {"shift_waypoints": 0.0, "lag_ms": 0.0, "lag_m": 0.0})

        # Constant-velocity backward extrapolation for indices before the window.
        velocity = ego[1] - ego[0]
        idx = np.arange(n) - shift
        out = np.where(
            idx[:, None] >= 0,
            ego[np.clip(idx, 0, n - 1)],
            ego[0] + idx[:, None] * velocity,
        )

        realized = {
            "shift_waypoints": float(shift),
            "lag_ms": float(shift * period_ms),
            "lag_m": float(np.linalg.norm(out[-1, [EGO_X, EGO_Y]] - ego[-1, [EGO_X, EGO_Y]])),
        }
        return PerturbationResult(sample.with_ego(out), realized)
