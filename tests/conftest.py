"""Shared fixtures. Everything here is synthetic -- the real split is gated."""

from __future__ import annotations

import numpy as np
import pytest

from edr.schema import EGO_DIMS, EGO_HISTORY_LEN, Sample

WAYPOINT_DT_S = 0.1
SPEED_MPS = 10.0


@pytest.fixture
def sample() -> Sample:
    """A straight-line ego history at constant speed along +x."""
    t = np.arange(EGO_HISTORY_LEN) * WAYPOINT_DT_S
    ego = np.zeros((EGO_HISTORY_LEN, EGO_DIMS))
    ego[:, 0] = SPEED_MPS * t
    return Sample(
        scenario_id="synthetic-0001",
        ego_history=ego,
        video_key="synthetic-0001/front",
        waypoint_dt_s=WAYPOINT_DT_S,
        speed_mps=SPEED_MPS,
    )
