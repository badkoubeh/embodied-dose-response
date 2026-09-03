"""Raw-artifact writer (PLAN.md 3).

STUB.

Writes the two-file split the architecture diagram specifies: `records/*.jsonl`
for the metadata and `trajectories/*.npz` for the float arrays, joined on
`RawRecord.trajectory_key`.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

from edr.schema import RawRecord


class RawRecordWriter:
    """Append-only writer for one shard's raw artifacts."""

    def __init__(self, out_dir: Path, shard_index: int) -> None:
        raise NotImplementedError

    def write(self, record: RawRecord) -> None:
        """Append one JSONL line and stage its trajectory for the npz."""
        raise NotImplementedError

    def __enter__(self) -> RawRecordWriter:
        raise NotImplementedError

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        raise NotImplementedError
