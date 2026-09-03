"""Determinism of the per-cell RNG (PLAN.md 2.4)."""

from __future__ import annotations

import os
import subprocess
import sys

from edr.seeding import cell_rng, cell_seed

CELL = ("scenario-0042", "ego_drift", 3, 1)


def test_seed_is_deterministic_within_a_process():
    assert cell_seed(*CELL) == cell_seed(*CELL)


def test_seed_is_stable_across_processes():
    """The assertion that catches `hash()`.

    Python salts `hash()` on str per process, so a seeding scheme built on it
    passes every in-process test and still produces shards that disagree. Run
    the hash in fresh interpreters with different PYTHONHASHSEED values and
    require identical output.
    """
    code = "from edr.seeding import cell_seed; print(cell_seed('scenario-0042','ego_drift',3,1))"
    outs = set()
    for seed in (0, 1, 12345):
        env = {**os.environ, "PYTHONHASHSEED": str(seed)}
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
        )
        outs.add(proc.stdout.strip())
    assert len(outs) == 1, f"seed varies with PYTHONHASHSEED: {outs}"
    assert outs.pop() == str(cell_seed(*CELL))


def test_distinct_cells_get_distinct_seeds():
    seeds = {
        cell_seed(f"scenario-{s:04d}", axis, level, seed_idx)
        for s in range(20)
        for axis in ("ego_noise", "ego_drift", "staleness")
        for level in range(9)
        for seed_idx in range(2)
    }
    assert len(seeds) == 20 * 3 * 9 * 2


def test_delimiter_prevents_field_collisions():
    """('a|b', 'c') and ('a', 'b|c') must not collapse to the same seed."""
    assert cell_seed("a|b", "c", 0, 0) != cell_seed("a", "b|c", 0, 0)


def test_rng_stream_is_reproducible():
    a = cell_rng(*CELL).normal(size=32)
    b = cell_rng(*CELL).normal(size=32)
    assert (a == b).all()


def test_rng_differs_between_seed_indices():
    a = cell_rng("s", "ego_noise", 0, 0).normal(size=32)
    b = cell_rng("s", "ego_noise", 0, 1).normal(size=32)
    assert not (a == b).all()


# Golden values. If a change to `cell_seed` moves these, every record already
# emitted becomes irreproducible while still looking valid -- the failure this
# test exists to make loud. Changing the construction means bumping a scheme
# version, not editing these numbers.
GOLDEN_SEEDS = {
    ("scn_000123", "ego_noise", 3, 1): 6441390019138447049,
    ("scn_000123", "ego_noise", 3, 0): 16673271733013234777,
    ("scn_000123", "ego_drift", 3, 1): 7226927825817425755,
    ("scn_000123", "ego_noise", 4, 1): 16237412566835830390,
    ("scn_000124", "ego_noise", 3, 1): 9215411993179240308,
}


def test_golden_seed_values():
    for key, expected in GOLDEN_SEEDS.items():
        assert cell_seed(*key) == expected, f"seeding scheme changed for {key}"


def test_shard_order_independence():
    """PLAN.md 2.4: results must not depend on how cells were distributed."""
    import random

    keys = [
        (f"scenario-{s:04d}", axis, level, seed_idx)
        for s in range(30)
        for axis in ("ego_noise", "ego_drift", "staleness")
        for level in range(9)
        for seed_idx in range(2)
    ]
    in_order = {k: cell_seed(*k) for k in keys}
    shuffled = keys[:]
    random.Random(0).shuffle(shuffled)
    assert {k: cell_seed(*k) for k in shuffled} == in_order


def test_seed_ignores_model():
    """The confounded secondary arm (PLAN.md 1) is a PAIRED comparison: both
    models must see bit-identical perturbation draws, so `model_id` is
    deliberately absent from the key. Adding it would quietly destroy that."""
    import inspect

    assert "model" not in inspect.signature(cell_seed).parameters
