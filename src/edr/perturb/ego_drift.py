"""Dead-reckoning drift on the ego-state history (PLAN.md 3.2, `ego_drift`).

PLAN.md 3.2 calls this the differentiator axis. It models what happens when
absolute positioning drops out -- a tunnel, an urban canyon, a jammed receiver --
and the pose estimate falls back on dead reckoning, accumulating a *correlated*
error that grows with distance travelled. That correlation is the whole point:
`ego_noise` jitters each waypoint independently and averages out, while drift
displaces the entire trajectory coherently.

Anchored to the wheel-mounted-IMU field study PLAN.md 3.2 cites: 0.47 % of
distance travelled and 1.13 deg heading RMS over 26 km.

**Why `outage_duration_s` exists.** Severity is %/distance, but the history window
is only ~1.6 s long -- at 10 m/s that is 16 m, over which even 2 %/distance is
0.32 m. Drift that small would make this axis inert, and the axis would be
measuring the wrong thing. The error that matters accumulated *before* the
window opened: PLAN.md 3.2 says each level should map to a named scenario such as
"tunnel GNSS outage at 30 s", which presumes an outage duration without naming
one. It is an explicit parameter here, and it is what converts %/distance into
metres. See the open issue.
"""

from __future__ import annotations

import numpy as np

from edr.perturb.base import PerturbationResult, register
from edr.schema import EGO_X, EGO_Y, EGO_YAW, Sample


@register
class EgoDrift:
    """Constant bias plus integrated random walk on position and heading.

    Severity is drift as a percentage of distance travelled. The offset at each
    waypoint is proportional to the distance dead-reckoned up to that waypoint,
    so the perturbation is coherent along the path and largest at the current
    pose -- which is the pose the policy conditions on most strongly.
    """

    name = "ego_drift"
    unit = "pct_distance"

    def __init__(
        self,
        outage_duration_s: float = 30.0,
        heading_deg_per_pct: float = 2.4,
        random_walk_frac: float = 0.25,
    ) -> None:
        # heading_deg_per_pct defaults to the field-study ratio: 1.13 deg of
        # heading RMS accompanying 0.47 %/distance of position drift.
        if outage_duration_s <= 0:
            raise ValueError("outage_duration_s must be positive")
        if random_walk_frac < 0:
            raise ValueError("random_walk_frac must be non-negative")
        self.outage_duration_s = float(outage_duration_s)
        self.heading_deg_per_pct = float(heading_deg_per_pct)
        self.random_walk_frac = float(random_walk_frac)

    def apply(
        self, sample: Sample, severity: float, rng: np.random.Generator
    ) -> PerturbationResult:
        if severity < 0:
            raise ValueError(f"severity must be non-negative, got {severity}")
        if severity == 0.0:
            return PerturbationResult(sample, {"pos_offset_m": 0.0, "yaw_offset_deg": 0.0})

        ego = sample.ego_history
        n = ego.shape[0]
        frac = severity / 100.0

        # Distance dead-reckoned by each waypoint. The window sits at the END of
        # the outage, so the oldest waypoint has already accumulated nearly the
        # full outage distance and the newest has accumulated all of it.
        d_window = sample.speed_mps * sample.waypoint_dt_s * np.arange(n)
        d_total = sample.speed_mps * self.outage_duration_s
        dist = d_total - (d_window[-1] - d_window)
        dist = np.maximum(dist, 0.0)

        # Constant bias: one direction for the whole outage, drawn per cell.
        theta = rng.uniform(0.0, 2.0 * np.pi)
        bias_dir = np.array([np.cos(theta), np.sin(theta)])
        yaw_sign = 1.0 if rng.random() < 0.5 else -1.0

        pos = np.outer(frac * dist, bias_dir)
        yaw = yaw_sign * np.deg2rad(severity * self.heading_deg_per_pct) * (dist / dist[-1])

        # Integrated random walk on top, scaled so its RMS at the current pose is
        # `random_walk_frac` of the bias magnitude there.
        if self.random_walk_frac > 0:
            steps = rng.normal(0.0, 1.0, size=(n, 2))
            walk = np.cumsum(steps, axis=0)
            walk_rms = float(np.sqrt(np.mean(np.sum(walk[-1:] ** 2, axis=1))))
            if walk_rms > 0:
                pos = pos + walk * (self.random_walk_frac * frac * dist[-1] / walk_rms)

        out = ego.copy()
        out[:, [EGO_X, EGO_Y]] += pos
        out[:, EGO_YAW] += yaw

        realized = {
            "pos_offset_m": float(np.linalg.norm(pos[-1])),
            "yaw_offset_deg": float(np.rad2deg(abs(yaw[-1]))),
            "outage_distance_m": float(d_total),
        }
        return PerturbationResult(sample.with_ego(out), realized)
