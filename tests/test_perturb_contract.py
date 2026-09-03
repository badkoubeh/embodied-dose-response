"""The contract every axis must satisfy (edr.perturb.base.Perturbation).

Parametrized over the registry, so a fourth axis added later is covered the
moment it registers.
"""

from __future__ import annotations

import numpy as np
import pytest

from edr.perturb import Perturbation, available, build
from edr.schema import EGO_X, EGO_Y, Sample
from edr.seeding import cell_rng

# A mid-grid severity per axis, in that axis's own unit.
MID_SEVERITY = {"ego_noise": 0.5, "ego_drift": 1.0, "staleness": 500.0}
AXES = sorted(MID_SEVERITY)


def _displacement(before: Sample, after: Sample) -> float:
    """RMS position displacement over the history, in metres."""
    delta = after.ego_history[:, [EGO_X, EGO_Y]] - before.ego_history[:, [EGO_X, EGO_Y]]
    return float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))


def test_registry_covers_the_three_planned_axes():
    assert available() == AXES


@pytest.mark.parametrize("axis", AXES)
def test_registry_resolves_from_a_config_string(axis):
    p = build(axis)
    assert isinstance(p, Perturbation)
    assert p.name == axis
    assert p.unit


def test_unknown_axis_is_rejected():
    with pytest.raises(KeyError, match="unknown perturbation"):
        build("ego_teleport")


@pytest.mark.parametrize("axis", AXES)
def test_zero_severity_is_an_exact_identity(axis, sample):
    out = build(axis).apply(sample, 0.0, cell_rng("s", axis, 0, 0))
    assert np.array_equal(out.sample.ego_history, sample.ego_history)


@pytest.mark.parametrize("axis", AXES)
def test_negative_severity_is_rejected(axis, sample):
    with pytest.raises(ValueError, match="non-negative"):
        build(axis).apply(sample, -1.0, cell_rng("s", axis, 0, 0))


@pytest.mark.parametrize("axis", AXES)
def test_same_seed_reproduces_bit_identically(axis, sample):
    p = build(axis)
    sev = MID_SEVERITY[axis]
    a = p.apply(sample, sev, cell_rng("s", axis, 4, 0))
    b = p.apply(sample, sev, cell_rng("s", axis, 4, 0))
    assert np.array_equal(a.sample.ego_history, b.sample.ego_history)
    assert a.realized == b.realized


@pytest.mark.parametrize("axis", AXES)
def test_input_sample_is_never_mutated(axis, sample):
    original = sample.ego_history.copy()
    build(axis).apply(sample, MID_SEVERITY[axis], cell_rng("s", axis, 4, 0))
    assert np.array_equal(sample.ego_history, original)


@pytest.mark.parametrize("axis", AXES)
def test_video_key_is_untouched(axis, sample):
    """All three axes are ego-state degradations, so one decode serves the whole
    sweep (PLAN.md 2.5). This includes `staleness`, which is scoped to the ego
    channel deliberately."""
    out = build(axis).apply(sample, MID_SEVERITY[axis], cell_rng("s", axis, 4, 0))
    assert out.sample.video_key == sample.video_key


@pytest.mark.parametrize("axis", AXES)
def test_magnitude_is_monotone_in_severity(axis, sample):
    """Exactly monotone along a fixed rng path, not merely monotone in expectation.

    Every axis draws at unit scale and multiplies by severity, so holding the rng
    path fixed makes the displacement a non-decreasing function of severity
    pathwise. That is a much sharper assertion than averaging over seeds and
    comparing with a tolerance -- and it is what makes adjacent grid levels
    common random numbers.
    """
    p = build(axis)
    grid = np.linspace(0.0, MID_SEVERITY[axis] * 4, 9)
    mags = [
        _displacement(sample, p.apply(sample, sev, cell_rng("s", axis, 0, 0)).sample)
        for sev in grid
    ]
    assert all(b >= a for a, b in zip(mags, mags[1:], strict=False)), mags
    assert mags[-1] > mags[1] > 0.0


@pytest.mark.parametrize("axis", AXES)
def test_realized_values_are_reported_with_units(axis, sample):
    """PLAN.md 2.2: the physical value gets logged, and for a stochastic draw the
    nominal severity is not that value."""
    out = build(axis).apply(sample, MID_SEVERITY[axis], cell_rng("s", axis, 4, 0))
    assert out.realized
    assert all(isinstance(v, float) for v in out.realized.values())
    assert all(k.rsplit("_", 1)[-1] for k in out.realized)


# Golden outputs. These pin the ORDER AND COUNT of rng draws inside each
# `apply`, which the seed alone does not: inserting or reordering a draw changes
# every output while the seed stays the same, silently invalidating already
# emitted records. See `edr.perturb.base.Perturbation.apply`.
GOLDEN_FINAL_POSE = {
    "ego_noise": [14.74886520629, 0.927078724167, -0.000237363157],
    "ego_drift": [11.597892426115, 1.44412784394, -0.041887902048],
    "staleness": [10.0, 0.0, 0.0],
}


@pytest.mark.parametrize("axis", AXES)
def test_golden_output_pins_draw_order(axis, sample):
    golden_sample = Sample(
        scenario_id="golden",
        ego_history=sample.ego_history,
        video_key="golden/front",
        waypoint_dt_s=sample.waypoint_dt_s,
        speed_mps=sample.speed_mps,
    )
    out = build(axis).apply(golden_sample, MID_SEVERITY[axis], cell_rng("golden", axis, 4, 0))
    assert out.sample.ego_history[-1] == pytest.approx(GOLDEN_FINAL_POSE[axis], abs=1e-9)
