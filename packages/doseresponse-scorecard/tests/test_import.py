"""The package must stand alone -- it is published separately and shared."""

from __future__ import annotations

import subprocess
import sys


def test_imports_without_edr_on_the_path():
    code = (
        "import sys;"
        "import doseresponse_scorecard as d;"
        "assert 'edr' not in sys.modules;"
        "print(d.__version__)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip()


def test_public_api_surface_is_present():
    import doseresponse_scorecard.fitting as fitting
    import doseresponse_scorecard.intervals as intervals
    import doseresponse_scorecard.thresholds as thresholds

    assert thresholds.Direction.LOWER_IS_BETTER
    assert callable(fitting.fit_glm)
    assert callable(intervals.fieller_ratio_ci)
