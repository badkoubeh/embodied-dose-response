"""Binary safety predicates (PLAN.md 3.3).

STUB -- needs map and actor geometry from the dataset.

Open-loop predicates only: these ask whether the *predicted* trajectory would
have collided or left the drivable surface, which is not the same as what would
happen if the policy actually drove. Same weak-proxy caveat as `trajectory`.
"""

from __future__ import annotations

import numpy as np


def collision(trajectory: np.ndarray, scene: object) -> bool:
    """Whether the predicted trajectory intersects another actor."""
    raise NotImplementedError("blocked on gated dataset access (PLAN.md 4)")


def offroad(trajectory: np.ndarray, scene: object) -> bool:
    """Whether the predicted trajectory leaves the drivable surface."""
    raise NotImplementedError("blocked on gated dataset access (PLAN.md 4)")
