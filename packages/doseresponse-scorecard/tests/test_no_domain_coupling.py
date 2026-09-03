"""PLAN.md 3.2: "Nothing Alpamayo-specific enters this package."

That promise decays under deadline pressure, so it fails CI instead. The guard
targets terms that could only appear if the domain leaked in -- not generic
prose, which may legitimately explain what domain-neutrality means.
"""

from __future__ import annotations

import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "doseresponse_scorecard"

FORBIDDEN = [
    "alpamayo",
    "nvidia",
    "cosmos",
    "waypoint",
    "minade",
    "scenario_id",
    "ego_",
    "import edr",
    "from edr",
]


@pytest.mark.parametrize("term", FORBIDDEN)
def test_no_domain_specific_terms(term):
    hits = [
        f"{path.name}:{i}"
        for path in SRC.rglob("*.py")
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if term in line.lower()
    ]
    assert not hits, f"domain term {term!r} leaked into the shared package: {hits}"


def test_package_does_not_depend_on_edr():
    pyproject = (SRC.parents[1] / "pyproject.toml").read_text().lower()
    assert "edr" not in pyproject.split("[build-system]")[0]
