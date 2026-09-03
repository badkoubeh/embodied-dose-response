"""Dose-response curve fitting (PLAN.md 4).

STUB.

Wrap `statsmodels`; do not hand-roll a probit fitter. Validate against R's `drc`
(`ED`, `EDcomp`), with `pystatsbio.doseresponse` as the Python cross-check and
`rpy2` as the fallback if Python coverage proves thin.
"""

from __future__ import annotations

import numpy as np


def fit_glm(
    severity: np.ndarray,
    outcome: np.ndarray,
    link: str = "probit",
    cluster: np.ndarray | None = None,
    log_severity: bool = False,
) -> object:
    """Binary dose-response GLM.

    `cluster` gives cluster-robust (sandwich) standard errors. Pass it whenever
    the same unit is measured at more than one severity -- otherwise repeated
    measures are treated as independent and every downstream interval is too
    narrow.

    `log_severity` per PLAN.md 4: log where the axis spans orders of magnitude,
    linear where it does not.
    """
    raise NotImplementedError


def fit_loglogistic(severity: np.ndarray, response: np.ndarray, monotone: bool = True) -> object:
    """4-parameter log-logistic fit for a continuous response."""
    raise NotImplementedError


def fit_isotonic(severity: np.ndarray, response: np.ndarray) -> object:
    """Monotone non-parametric fit, for when the link is suspect.

    A non-monotone response is either a real finding or undersampling. Resolve it
    by adding local samples, never by smoothing it away.
    """
    raise NotImplementedError
