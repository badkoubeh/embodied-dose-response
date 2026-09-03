"""Meta-action extraction and reasoning-action consistency (PLAN.md 3.2).

STUB -- blocked on the calibration set.

This is a REIMPLEMENTATION of the consistency reward in spirit, not NVIDIA's
computation: the released `aggregated_reward.py` ships ADE and comfort only. Its
precision and recall against hand labels are part of the result and must be
reported, per PLAN.md 6.

Two things the implementation must get right:

**Calibrate across severity, not just on clean records.** If reasoning text grows
hedged or malformed as severity rises, extractor precision falls, measured
consistency falls with it, and that is indistinguishable from the effect this
project claims to measure. Reporting precision as a function of severity, and
showing it flat, is a necessary validity check rather than a nice-to-have.

**The deadbands are free parameters that move the primary outcome.** `hold` and
`straight` are defined by thresholds on longitudinal acceleration and curvature;
both need calibrating against hand labels and a reported sensitivity analysis.
See the open issues.
"""

from __future__ import annotations

import numpy as np

from edr.schema import MetaAction, RawRecord

EXTRACTOR_VERSION = "0.0.0-unimplemented"


def extract_meta_action(reasoning_text: str) -> MetaAction | None:
    """Rule-based parse of the chain-of-causation into a meta-action.

    Returns None when no meta-action can be identified -- an abstention that must
    be counted and reported separately, never silently folded into "inconsistent".
    """
    raise NotImplementedError("blocked on the calibration set (PLAN.md 5, Stage 1)")


def derive_meta_action(
    trajectory: np.ndarray, accel_deadband: float, curvature_deadband: float
) -> MetaAction:
    """Meta-action from kinematics: sign of longitudinal acceleration and curvature."""
    raise NotImplementedError("blocked on the calibration set (PLAN.md 5, Stage 1)")


def consistency(record: RawRecord) -> bool:
    """Agreement between the reasoning-derived and trajectory-derived meta-actions.

    Report the longitudinal and lateral components separately as well; see
    `edr.schema.MetaAction` for why the comparison is a pair rather than one label.
    """
    raise NotImplementedError("blocked on the calibration set (PLAN.md 5, Stage 1)")
