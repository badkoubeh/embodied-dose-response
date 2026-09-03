"""Record schema invariants (PLAN.md 3.3)."""

from __future__ import annotations

import json
from dataclasses import fields, replace

import numpy as np
import pytest

from edr.schema import (
    TRAJ_DIMS,
    TRAJ_HORIZON,
    Lateral,
    Longitudinal,
    MetaAction,
    RawRecord,
    Sample,
)


def _record(k: int = 1) -> RawRecord:
    rng = np.random.default_rng(0)
    return RawRecord(
        scenario_id="scn_000123",
        axis="ego_drift",
        severity=0.47,
        unit="pct_distance",
        level_index=3,
        seed_index=1,
        rng_seed=7226927825817425755,
        trajectory=rng.normal(size=(k, TRAJ_HORIZON, TRAJ_DIMS)),
        reasoning_text="slowing for the vehicle ahead",
        latency_s=3.4,
        peak_vram_gb=23.1,
        git_sha="abc1234",
        weights_hash="sha256:deadbeef",
        config_hash="cfg:0001",
        model_id="nvidia/Alpamayo-1.5-10B",
        realized={"pos_offset_m": 1.41},
    )


@pytest.mark.parametrize("k", [1, 6])
def test_trajectory_accepts_any_k(k):
    """K=1 and K=6 are both valid, so resolving the minADE-6 question either way
    needs no schema migration. See the open issue."""
    assert _record(k).n_traj_samples == k


@pytest.mark.parametrize(
    "shape", [(TRAJ_HORIZON, TRAJ_DIMS), (1, 32, 3), (1, TRAJ_HORIZON, 2), (0, 64, 3)]
)
def test_bad_trajectory_shapes_are_rejected(shape):
    with pytest.raises(ValueError, match="trajectory"):
        replace(_record(), trajectory=np.zeros(shape))


def test_jsonl_omits_the_trajectory_and_keeps_a_join_key():
    """PLAN.md 3 splits storage: records/*.jsonl beside trajectories/*.npz."""
    rec = _record(6)
    d = json.loads(rec.to_jsonl())
    assert "trajectory" not in d
    assert d["trajectory_key"] == rec.trajectory_key
    assert d["n_traj_samples"] == 6


def test_jsonl_roundtrip_reproduces_the_record():
    rec = _record()
    back = RawRecord.from_json_dict(json.loads(rec.to_jsonl()), rec.trajectory)
    assert back.to_jsonl() == rec.to_jsonl()
    assert np.array_equal(back.trajectory, rec.trajectory)


def test_raw_record_carries_no_derived_quantity():
    """PLAN.md 2.1: the GPU plane emits raw artifacts only. A scoring field
    leaking in here is what would make recalibration cost a re-inference."""
    derived = {"consistent", "minade6", "ade", "fde", "collision", "offroad", "meta_action"}
    assert derived.isdisjoint({f.name for f in fields(RawRecord)})


def test_raw_record_field_names_are_pinned():
    """Schema drift guard: these names are the on-disk format."""
    assert {f.name for f in fields(RawRecord)} == {
        "scenario_id",
        "axis",
        "severity",
        "unit",
        "level_index",
        "seed_index",
        "rng_seed",
        "trajectory",
        "reasoning_text",
        "latency_s",
        "peak_vram_gb",
        "git_sha",
        "weights_hash",
        "config_hash",
        "model_id",
        "realized",
    }


def test_level_index_is_present_for_reproducibility():
    """PLAN.md 2.4 seeds from `level_index`, but PLAN.md 3.3 lists only the
    severity float. Without the index, reproducing a cell means reverse-mapping a
    float to a grid position, which breaks on any formatting change."""
    assert "level_index" in {f.name for f in fields(RawRecord)}


def test_records_holding_arrays_do_not_use_generated_eq():
    """A frozen dataclass __eq__ over an ndarray returns an array, and bool() on
    it raises. Every record type here opts out."""
    for cls in (Sample, RawRecord):
        assert "__eq__" not in cls.__dict__


def test_meta_action_is_a_pair_and_compares_componentwise():
    a = MetaAction(Longitudinal.DECELERATE, Lateral.STRAIGHT)
    assert a.agrees_with(MetaAction(Longitudinal.DECELERATE, Lateral.STRAIGHT))
    assert not a.agrees_with(MetaAction(Longitudinal.DECELERATE, Lateral.LEFT))
    assert not a.agrees_with(MetaAction(Longitudinal.HOLD, Lateral.STRAIGHT))


def test_sample_rejects_wrong_ego_shape():
    with pytest.raises(ValueError, match="ego_history"):
        Sample("s", np.zeros((8, 3)), "v", 0.1, 10.0)
