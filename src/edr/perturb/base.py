"""Perturbation interface and registry (PLAN.md 3.2).

One interface, three implementations, resolved by name so Hydra configs stay
declarative.

Two deliberate deviations from the protocol sketched in PLAN.md 3.2:

**`levels()` is not on the interface.** PLAN.md 3.2 puts
`levels(grid: str) -> list[float]` on the perturbation object, but PLAN.md 3.1
also defines a `grid/` config group, and PLAN.md 2.2 wants physical units at the
boundary. Holding severity values in `configs/axis/*.yaml` instead means a
perturbation contains no numbers at all -- it only knows how to *apply* one --
so re-anchoring a grid after the Stage-2 pilot is a config edit, never a code
change. See `edr.perturb.levels`.

**`apply()` returns `PerturbationResult`, not a bare `Sample`.** For a stochastic
axis the nominal severity is not what was actually drawn: asking for sigma = 0.5 m
over 16 waypoints yields some realized RMS near but not equal to it. PLAN.md 2.2
says the physical value is what gets logged, and the realized draw is that value.
It rides along to `RawRecord.realized`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from edr.schema import Sample


@dataclass(frozen=True, slots=True)
class PerturbationResult:
    """A perturbed sample plus what the draw actually produced.

    `realized` is free-form per axis (e.g. `{"pos_rms_m": 0.48, "yaw_rms_deg": 1.1}`)
    and is logged verbatim. Keys should carry their unit as a suffix.
    """

    sample: Sample
    realized: dict[str, float]


@runtime_checkable
class Perturbation(Protocol):
    """A physically-grounded, benign degradation of one input channel."""

    name: str
    unit: str

    def apply(
        self, sample: Sample, severity: float, rng: np.random.Generator
    ) -> PerturbationResult:
        """Apply `severity` (in `self.unit`) to `sample`.

        Contract, enforced by `tests/test_perturb_contract.py`:

        * `severity == 0` returns the sample unchanged, exactly.
        * Deterministic given `rng` state: same seed and severity, same output.
        * Never mutates `sample` or any array it holds.
        * Perturbation magnitude is monotone non-decreasing in `severity`.

        One contract that is invisible in the signature: **the order and count of
        `rng` draws is part of the reproducibility guarantee.** Adding a draw or
        reordering two of them changes every output while the seed stays the
        same, silently invalidating already-emitted records. Implementations draw
        at unit scale and multiply by severity, in a documented order, and
        `tests/test_perturb_contract.py` pins the resulting arrays against golden
        values so a reordering fails loudly.
        """
        ...


# --- Registry --------------------------------------------------------------

_REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    """Class decorator. Registers under `cls.name` so configs can name it."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must define a non-empty class attribute `name`")
    if not getattr(cls, "unit", None):
        raise ValueError(f"{cls.__name__} must define a non-empty class attribute `unit`")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"perturbation {name!r} already registered by {_REGISTRY[name].__name__}")
    _REGISTRY[name] = cls
    return cls


def build(name: str, **kwargs: object) -> Perturbation:
    """Construct the registered perturbation `name` with axis-specific kwargs."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown perturbation {name!r}; available: {available()}") from None
    return cls(**kwargs)  # type: ignore[return-value]


def available() -> list[str]:
    """Registered axis names, sorted."""
    return sorted(_REGISTRY)
