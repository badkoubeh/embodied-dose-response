"""Decoded-video cache (PLAN.md 2.5).

STUB -- the dataset is gated.

The point of this cache is that no ego-state axis touches pixels: one decode per
scenario serves an entire severity sweep across all three axes and both seeds.
`Sample` holds a `video_key` rather than a tensor precisely so this stays true.
"""

from __future__ import annotations

from pathlib import Path


class VideoCache:
    """Decode once per scenario, reuse across every severity level and seed."""

    def __init__(self, root: Path, max_items: int | None = None) -> None:
        raise NotImplementedError("blocked on gated dataset access (PLAN.md 4)")

    def get(self, video_key: str) -> object:
        """Return the decoded video tensor for `video_key`."""
        raise NotImplementedError("blocked on gated dataset access (PLAN.md 4)")
