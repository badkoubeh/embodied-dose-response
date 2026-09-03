"""Deterministic per-cell RNG (PLAN.md 2.4).

Every cell's randomness is derived from its identity, not from a stream. That
buys two things the sweep depends on:

  * **Shard-order independence.** Cells are distributed across independent
    SageMaker jobs. Nothing may depend on which shard a cell landed in or on how
    many cells preceded it.
  * **Single-cell reproducibility.** `scripts/reproduce_cell.py` must regenerate
    any one cell in isolation, months later, from its identifier alone.

Two decisions here are load-bearing and easy to get wrong:

**Python's builtin `hash()` cannot be used.** `hash()` on `str` is salted per
process (PYTHONHASHSEED), so it returns different values in different jobs. It
would silently produce a sweep that cannot be reproduced and whose shards
disagree. `hashlib.blake2b` is stable across processes, machines, and releases.

**`model_id` is deliberately excluded from the hash.** PLAN.md 2.4 lists the key
as `(scenario_id, axis, level_index, seed_index)` without saying why the model is
absent, but it is the reason the confounded secondary arm (PLAN.md 1) is a
*paired* comparison: Alpamayo-1.5 and Alpamayo-R1 see bit-identical perturbation
draws at every cell, so any difference between them cannot be a difference in
noise realization. Adding `model_id` here would quietly destroy that.
"""

from __future__ import annotations

import hashlib

import numpy as np

# Unit separator. A delimiter that cannot occur in an identifier, so
# ("a|b", "c") and ("a", "b|c") cannot collide into the same seed.
_SEP = "\x1f"

# blake2b personalization: domain-separates these digests from any other hash
# in the project, at zero cost.
_PERSON = b"edr-cell-seed"


def cell_seed(scenario_id: str, axis: str, level_index: int, seed_index: int) -> int:
    """Stable 64-bit seed for one cell.

    Deterministic across processes, machines, and Python versions. Note this
    takes `level_index`, not the severity float: grid indices are exact, and
    keying on a float would make the seed hostage to formatting and rounding.
    """
    payload = _SEP.join((scenario_id, axis, str(int(level_index)), str(int(seed_index))))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8, person=_PERSON).digest()
    return int.from_bytes(digest, "big")


def cell_rng(scenario_id: str, axis: str, level_index: int, seed_index: int) -> np.random.Generator:
    """The `Generator` for one cell. Same arguments always give the same stream."""
    return np.random.default_rng(cell_seed(scenario_id, axis, level_index, seed_index))
