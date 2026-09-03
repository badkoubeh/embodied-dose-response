"""Record schema for the dose-response sweep (PLAN.md 3.3).

Three record types, one per stage of the pipeline, plus the input `Sample` that
PLAN.md 3.2 references but never defines.

The split matters. PLAN.md 2.1 makes separating the GPU plane from the analysis
plane the single most important architectural decision here: GPU jobs emit
`RawRecord`s and nothing else, and every derived quantity lives on `ScoredRecord`,
recomputed post-hoc on CPU. Recalibrating the meta-action extractor must never
cost a re-inference, which means no scoring field may ever leak into `RawRecord`.

Storage follows the same split (see the diagram in PLAN.md 3):

    records/*.jsonl        one line per RawRecord/ScoredRecord, trajectory elided
    trajectories/*.npz     the float arrays, keyed by `trajectory_key`

`to_json_dict()` deliberately omits the trajectory array; `trajectory_key` is the
join key back into the npz. Keeping metre-scale floats out of JSON also avoids a
lossy text round-trip on the primary measurement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from enum import StrEnum
from typing import Any

import numpy as np

# --- Array shape constants -------------------------------------------------
#
# NOTE (unverified): the ego-state layout below is inferred, not confirmed. The
# `PhysicalAI-Autonomous-Vehicles` split is gated and unavailable at scaffold
# time. PLAN.md 3.2 says "16-waypoint translation/rotation history" and PLAN.md
# 3.3 says `trajectory[64,3]`, which is consistent with planar (x, y, yaw) but
# does not pin it. Confirm against a real sample in Stage 0 before trusting any
# physical severity number, and see the open issue on waypoint spacing --
# PLAN.md gives 16 waypoints and a 100 ms *frame* period without saying the two
# match, and every severity value in configs/axis/ depends on which it is.

EGO_HISTORY_LEN = 16
EGO_DIMS = 3  # (x_m, y_m, yaw_rad)
TRAJ_HORIZON = 64
TRAJ_DIMS = 3  # (x_m, y_m, yaw_rad)

EGO_X, EGO_Y, EGO_YAW = 0, 1, 2


# --- Meta-actions ----------------------------------------------------------


class Longitudinal(StrEnum):
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    HOLD = "hold"


class Lateral(StrEnum):
    LEFT = "left"
    STRAIGHT = "straight"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class MetaAction:
    """A meta-action as a (longitudinal, lateral) pair rather than one flat label.

    PLAN.md 3.2 gives the flat set {accelerate, decelerate, hold, turn-left,
    turn-right, yield} for `extract_meta_action`, but `derive_meta_action` works
    from "sign of longitudinal acceleration and curvature" -- which yields two
    independent components, and which can never produce "yield" (a right-of-way
    concept, kinematically indistinguishable from decelerate). Comparing a
    6-way text label against a 2-tuple derived from kinematics is comparing
    across mismatched label spaces.

    Representing both sides as the same pair makes `consistency` well defined and
    lets the two components be reported separately, which is strictly more
    informative and costs nothing. Text-side "yield" maps to DECELERATE; whether
    that collapse is acceptable is an open question on the tracker.
    """

    longitudinal: Longitudinal
    lateral: Lateral

    def agrees_with(self, other: MetaAction) -> bool:
        """Conjunction of both components. Report components separately too."""
        return self.longitudinal is other.longitudinal and self.lateral is other.lateral


# --- Pipeline records ------------------------------------------------------


# NOTE on `eq=False` below: a frozen dataclass generates an `__eq__` that
# evaluates `self.trajectory == other.trajectory`, which returns an ARRAY, and
# `bool()` on that raises "truth value of an array is ambiguous". Every record
# type here holds an ndarray, so all of them opt out and fall back to identity
# comparison. Compare records field by field, or compare arrays with
# `np.array_equal`.


@dataclass(frozen=True, slots=True, eq=False)
class Sample:
    """One scenario as handed to a `Perturbation` (PLAN.md 3.2).

    Holds the ego-state that perturbations act on, plus a *reference* to the
    video rather than the decoded tensor. That indirection is what makes
    PLAN.md 2.5 ("cache the pixels") work: every ego-state axis leaves
    `video_key` untouched, so one decode serves an entire severity sweep.

    `ego_history` is ordered oldest-first; index -1 is the current pose.
    """

    scenario_id: str
    ego_history: np.ndarray  # (EGO_HISTORY_LEN, EGO_DIMS)
    video_key: str
    waypoint_dt_s: float
    speed_mps: float
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = (EGO_HISTORY_LEN, EGO_DIMS)
        if self.ego_history.shape != expected:
            raise ValueError(f"ego_history must be {expected}, got {self.ego_history.shape}")

    def with_ego(self, ego_history: np.ndarray) -> Sample:
        """Return a copy carrying new ego-state. Never mutates in place."""
        return replace(self, ego_history=ego_history)


@dataclass(frozen=True, slots=True, eq=False)
class RawRecord:
    """GPU-plane output. Immutable, and carries no derived quantity whatsoever.

    `trajectory` is (K, TRAJ_HORIZON, TRAJ_DIMS). K is the number of sampled
    trajectories and K=1 is valid and the default.

    On the leading axis: PLAN.md 3.2 pins the runner to `num_traj_samples=1`
    while PLAN.md 3.2 metrics ask for `minade(record, k=6)` and PLAN.md 4
    calibrates against NVIDIA's published minADE-6 of 1.22 m. Those cannot all
    hold, and the resolution is a live question on the tracker. Carrying an
    explicit K here means either answer lands without a schema migration.

    `realized` records what the perturbation *actually* drew, which for a
    stochastic axis is not the nominal `severity`. PLAN.md 2.2 wants physical
    values logged; the nominal level alone does not satisfy that.
    """

    scenario_id: str
    axis: str
    severity: float
    unit: str
    level_index: int
    seed_index: int
    rng_seed: int
    trajectory: np.ndarray
    reasoning_text: str
    latency_s: float
    peak_vram_gb: float
    git_sha: str
    weights_hash: str
    config_hash: str
    model_id: str
    realized: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = self.trajectory.shape
        if len(shape) != 3 or shape[1:] != (TRAJ_HORIZON, TRAJ_DIMS):
            raise ValueError(f"trajectory must be (K, {TRAJ_HORIZON}, {TRAJ_DIMS}), got {shape}")
        if shape[0] < 1:
            raise ValueError(f"trajectory needs K >= 1, got K={shape[0]}")

    @property
    def n_traj_samples(self) -> int:
        return int(self.trajectory.shape[0])

    @property
    def trajectory_key(self) -> str:
        """Join key into trajectories/*.npz. Unique per cell by construction."""
        return (
            f"{self.model_id}|{self.scenario_id}|{self.axis}|{self.level_index}|{self.seed_index}"
        )

    def to_json_dict(self) -> dict[str, Any]:
        # Built field by field rather than via `asdict`, which would deepcopy the
        # trajectory array only for it to be discarded -- once per record, for
        # every record in the sweep.
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "trajectory"}
        d["realized"] = dict(self.realized)
        d["trajectory_key"] = self.trajectory_key
        d["n_traj_samples"] = self.n_traj_samples
        return d

    def to_jsonl(self) -> str:
        return json.dumps(self.to_json_dict(), sort_keys=True)

    @classmethod
    def from_json_dict(cls, d: dict[str, Any], trajectory: np.ndarray) -> RawRecord:
        """Rehydrate. `trajectory` comes from the npz under `trajectory_key`."""
        fields = {k: v for k, v in d.items() if k not in ("trajectory_key", "n_traj_samples")}
        return cls(trajectory=trajectory, **fields)


@dataclass(frozen=True, slots=True, eq=False)
class ScoredRecord:
    """Analysis-plane output: a RawRecord plus everything derived from it.

    Recomputable from `raw` alone. `extractor_version` exists so a scorecard can
    state which calibration of the meta-action extractor produced it -- the
    extractor is a reimplementation of a reward NVIDIA did not release, so its
    version is part of the result, not an implementation detail.
    """

    raw: RawRecord
    meta_action_text: MetaAction | None
    meta_action_traj: MetaAction
    consistent: bool
    consistent_longitudinal: bool
    consistent_lateral: bool
    ade: float
    fde: float
    collision: bool
    offroad: bool
    extractor_version: str

    def to_json_dict(self) -> dict[str, Any]:
        d = self.raw.to_json_dict()
        d.update(
            meta_action_text=(
                None
                if self.meta_action_text is None
                else [
                    str(self.meta_action_text.longitudinal),
                    str(self.meta_action_text.lateral),
                ]
            ),
            meta_action_traj=[
                str(self.meta_action_traj.longitudinal),
                str(self.meta_action_traj.lateral),
            ],
            consistent=self.consistent,
            consistent_longitudinal=self.consistent_longitudinal,
            consistent_lateral=self.consistent_lateral,
            ade=self.ade,
            fde=self.fde,
            collision=self.collision,
            offroad=self.offroad,
            extractor_version=self.extractor_version,
        )
        return d

    def to_jsonl(self) -> str:
        return json.dumps(self.to_json_dict(), sort_keys=True)


@dataclass(frozen=True, slots=True)
class Cell:
    """Aggregate over scenarios at fixed (model, axis, severity, seed).

    Reporting and plotting only. **Do not fit on cell means.** The dose-response
    fit runs on `ScoredRecord`s with SEs clustered by `scenario_id`: the same
    scenarios recur at every severity level, so the observations are paired, and
    collapsing to cell means both discards that pairing and mis-weights the fit.
    See the open issue on repeated measures.

    `rate_ci_*` is a Clopper-Pearson interval on this cell's binomial and
    nothing else. PLAN.md 4 is explicit that it does not transfer to the
    interpolated threshold.
    """

    model_id: str
    axis: str
    severity: float
    unit: str
    seed_index: int
    n: int
    n_consistent: int
    rate: float
    rate_ci_low: float
    rate_ci_high: float
    ade_mean: float
    ade_se: float

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
