"""White noise on the ego-state history (PLAN.md 3.2, `ego_noise`).

Models an unbiased but noisy localization estimate: each waypoint's reported pose
carries independent error. The independence is what separates this axis from
`ego_drift`, where the error is correlated and accumulates.

**On the severity range.** PLAN.md 3.2 anchors this axis in IMU datasheets --
angular random walk from 0.05 deg/sqrt(hr) (ANELLO SiPhOG) to 0.29 deg/sqrt(hr)
(ADIS16485), bias instability from ~0.5 deg/hr to ~10 deg/hr (VN-100). Those are
*rates*, and the history window is short: 16 waypoints at a 100 ms period is
1.6 s, over which even 0.29 deg/sqrt(hr) integrates to about 0.006 deg of heading
noise. The entire datasheet range sits inside the regime where nothing happens.

So the datasheet figures label levels rather than defining the span. The quantity
that actually perturbs this input is the accumulated localization error of a
degraded or unaided estimator, which is why severity here is metres of position
error, spanning RTK-GNSS nominal to GNSS-denied urban canyon. See the open issue
on grid anchoring.
"""

from __future__ import annotations

import numpy as np

from edr.perturb.base import PerturbationResult, register
from edr.schema import EGO_X, EGO_Y, EGO_YAW, Sample


@register
class EgoNoise:
    """Gaussian additive noise on the waypoint translation and rotation history.

    Severity is the per-waypoint position standard deviation in metres. Heading
    noise is coupled to it through `yaw_sigma_deg_per_m` rather than dosed
    independently, so the axis stays one-dimensional and the grid stays a line.
    """

    name = "ego_noise"
    unit = "m"

    def __init__(self, yaw_sigma_deg_per_m: float = 2.0) -> None:
        # Provisional. PLAN.md does not pin the position/heading error ratio for
        # this axis, and it depends on the estimator being modelled. Re-anchor
        # after the Stage-2 pilot.
        if yaw_sigma_deg_per_m < 0:
            raise ValueError("yaw_sigma_deg_per_m must be non-negative")
        self.yaw_sigma_deg_per_m = float(yaw_sigma_deg_per_m)

    def apply(
        self, sample: Sample, severity: float, rng: np.random.Generator
    ) -> PerturbationResult:
        if severity < 0:
            raise ValueError(f"severity must be non-negative, got {severity}")
        if severity == 0.0:
            return PerturbationResult(sample, {"pos_rms_m": 0.0, "yaw_rms_deg": 0.0})

        ego = sample.ego_history
        yaw_sigma_rad = np.deg2rad(severity * self.yaw_sigma_deg_per_m)

        # Draw at UNIT scale, then multiply by severity. Distributionally
        # identical to drawing from N(0, severity^2), but it makes the
        # perturbation magnitude *exactly* monotone in severity along a fixed rng
        # path rather than monotone only in expectation. That turns the
        # monotonicity contract into an exact test instead of one with a
        # tolerance, and it couples adjacent grid levels into common random
        # numbers, which shaves variance along the dose axis for free.
        pos_noise = rng.normal(0.0, 1.0, size=(ego.shape[0], 2)) * severity
        yaw_noise = rng.normal(0.0, 1.0, size=ego.shape[0]) * yaw_sigma_rad

        out = ego.copy()
        out[:, [EGO_X, EGO_Y]] += pos_noise
        out[:, EGO_YAW] += yaw_noise

        realized = {
            "pos_rms_m": float(np.sqrt(np.mean(np.sum(pos_noise**2, axis=1)))),
            "yaw_rms_deg": float(np.rad2deg(np.sqrt(np.mean(yaw_noise**2)))),
        }
        return PerturbationResult(sample.with_ego(out), realized)
