"""Every Hydra config composes, and the invariants PLAN.md relies on hold."""

from __future__ import annotations

import itertools

import pytest
from hydra import compose, initialize

from edr.perturb import available, build, load_levels

AXES = ["ego_noise", "ego_drift", "staleness"]
GRIDS = ["pilot", "full"]
MODELS = ["alpamayo15", "alpamayo1"]
INFRAS = ["local", "sagemaker"]


def _cfg(**groups):
    overrides = [f"{k}={v}" for k, v in groups.items()]
    with initialize(version_base=None, config_path="../configs"):
        return compose(config_name="config", overrides=overrides)


@pytest.mark.parametrize(
    ("axis", "grid", "model", "infra"),
    list(itertools.product(AXES, GRIDS, MODELS, INFRAS)),
)
def test_every_combination_composes(axis, grid, model, infra):
    cfg = _cfg(axis=axis, grid=grid, model=model, infra=infra)
    assert cfg.axis.name == axis
    assert cfg.grid.name == grid


@pytest.mark.parametrize("axis", AXES)
def test_axis_target_resolves_in_the_registry(axis):
    cfg = _cfg(axis=axis)
    assert cfg.axis.target in available()
    build(cfg.axis.target, **dict(cfg.axis.params))


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("grid", GRIDS)
def test_levels_are_valid(axis, grid):
    """Sorted ascending, starting at the shared clean baseline, every level
    labelled with a physical referent (PLAN.md 2.2)."""
    levels = load_levels(_cfg(axis=axis).axis, grid)
    assert levels[0].is_clean
    assert [lv.index for lv in levels] == list(range(len(levels)))
    assert all(lv.unit and lv.label for lv in levels)


@pytest.mark.parametrize("axis", AXES)
def test_full_grid_size_matches_the_protocol(axis):
    """PLAN.md 4: 8-10 levels spanning clean to clearly-broken."""
    assert 8 <= len(load_levels(_cfg(axis=axis).axis, "full")) <= 10


@pytest.mark.parametrize("axis", AXES)
def test_axis_declares_a_unit_matching_its_implementation(axis):
    cfg = _cfg(axis=axis)
    assert cfg.axis.unit == build(cfg.axis.target, **dict(cfg.axis.params)).unit


def test_staleness_levels_are_frame_multiples():
    """PLAN.md 3.2 puts tau on multiples of the 100 ms frame period."""
    for lv in load_levels(_cfg(axis="staleness").axis, "full"):
        assert lv.value % 100 == 0, lv


def test_staleness_uses_one_seed():
    """The axis is deterministic given tau, so a second seed is a bit-identical
    duplicate. See the open issue on budget."""
    assert _cfg(axis="staleness").axis.seeds == 1


def test_alpamayo1_is_labelled_confounded():
    """PLAN.md 1 requires the confound label in code, in the scorecard, and in
    the paper. This is the code half, and it is the cheapest insurance here: it
    fails if anyone ever flips the flag or blanks the note."""
    cfg = _cfg(model="alpamayo1")
    assert cfg.model.confounded is True
    assert cfg.model.confound_note
    assert "NOT attributable" in cfg.model.confound_note


def test_primary_model_is_not_labelled_confounded():
    assert _cfg(model="alpamayo15").model.confounded is False


def test_only_the_pilot_grid_is_unreportable():
    """PLAN.md 2.3: adaptive dosing is permitted only in the throwaway pilot."""
    assert _cfg(grid="pilot").grid.reportable is False
    assert _cfg(grid="full").grid.reportable is True


def test_sagemaker_instance_has_headroom_over_the_weights():
    """Weights alone are 22.2 GB (PLAN.md 4). A 24 GB card leaves ~1.8 GB for
    activations, video tokens, and a long CoC KV cache. See the open issue."""
    assert _cfg(infra="sagemaker").infra.instance_type == "ml.g6e.xlarge"
