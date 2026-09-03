"""Perturbation axes (PLAN.md 3.2).

Importing this package registers all three axes, so `build(name)` resolves any
name a config can legally use.
"""

from edr.perturb.base import Perturbation, PerturbationResult, available, build, register
from edr.perturb.ego_drift import EgoDrift
from edr.perturb.ego_noise import EgoNoise
from edr.perturb.levels import Level, load_levels
from edr.perturb.staleness import Staleness

__all__ = [
    "EgoDrift",
    "EgoNoise",
    "Level",
    "Perturbation",
    "PerturbationResult",
    "Staleness",
    "available",
    "build",
    "load_levels",
    "register",
]
