"""Per-axis physical behaviour, beyond the shared contract."""

from __future__ import annotations

import numpy as np
import pytest

from edr.perturb import build
from edr.schema import EGO_X, EGO_Y, EGO_YAW
from edr.seeding import cell_rng


class TestEgoNoise:
    def test_realized_rms_tracks_nominal_sigma(self, sample):
        """Averaged over many draws, realized RMS should approach nominal sigma.

        The 2-D position RMS of an isotropic Gaussian with per-axis sigma is
        sigma * sqrt(2).
        """
        p = build("ego_noise")
        sigma = 0.5
        rms = np.mean(
            [
                p.apply(sample, sigma, cell_rng("s", "ego_noise", 5, k)).realized["pos_rms_m"]
                for k in range(200)
            ]
        )
        assert rms == pytest.approx(sigma * np.sqrt(2), rel=0.05)

    def test_noise_is_independent_across_waypoints(self, sample):
        """White, not correlated -- this is what separates the axis from drift.

        Averaged over draws, the lag-1 autocorrelation of the injected error
        should sit at zero.
        """
        p = build("ego_noise")
        corrs = []
        for k in range(300):
            out = p.apply(sample, 1.0, cell_rng("s", "ego_noise", 5, k))
            err = out.sample.ego_history[:, EGO_X] - sample.ego_history[:, EGO_X]
            corrs.append(np.corrcoef(err[:-1], err[1:])[0, 1])
        assert abs(float(np.mean(corrs))) < 0.1

    def test_heading_noise_is_coupled_to_position_noise(self, sample):
        p = build("ego_noise", yaw_sigma_deg_per_m=4.0)
        out = p.apply(sample, 0.5, cell_rng("s", "ego_noise", 5, 0))
        assert out.realized["yaw_rms_deg"] > 0
        flat = build("ego_noise", yaw_sigma_deg_per_m=0.0)
        out_flat = flat.apply(sample, 0.5, cell_rng("s", "ego_noise", 5, 0))
        assert out_flat.realized["yaw_rms_deg"] == 0.0


class TestEgoDrift:
    def test_offset_is_coherent_not_white(self, sample):
        """Drift displaces the whole path coherently; noise jitters it independently.

        This is the property that makes the two axes measure different things, so
        it is asserted as a comparison rather than against a bare threshold: the
        lag-1 autocorrelation of drift's injected error must sit far above
        `ego_noise`'s on the same data. With the random walk switched off, drift
        reduces to a pure ramp and the correlation is exactly 1.
        """

        def lag1(axis, **kw):
            out = build(axis, **kw).apply(sample, 1.0, cell_rng("s", axis, 5, 0))
            err = out.sample.ego_history[:, EGO_X] - sample.ego_history[:, EGO_X]
            return float(np.corrcoef(err[:-1], err[1:])[0, 1])

        drift, noise = lag1("ego_drift"), lag1("ego_noise")
        assert drift > 0.8
        assert drift - noise > 0.5, (drift, noise)
        assert lag1("ego_drift", random_walk_frac=0.0) == pytest.approx(1.0)

    def test_offset_grows_with_outage_duration(self, sample):
        """The reason `outage_duration_s` exists: %/distance alone cannot set the
        magnitude, because the history window is only ~16 m long."""
        offsets = [
            build("ego_drift", outage_duration_s=t)
            .apply(sample, 1.0, cell_rng("s", "ego_drift", 5, 0))
            .realized["pos_offset_m"]
            for t in (5.0, 15.0, 30.0, 60.0)
        ]
        assert all(b > a for a, b in zip(offsets, offsets[1:], strict=False))

    def test_offset_matches_the_physical_model(self, sample):
        """0.47 %/distance over a 30 s outage at 10 m/s is 300 m * 0.0047 = 1.41 m."""
        p = build("ego_drift", outage_duration_s=30.0, random_walk_frac=0.0)
        out = p.apply(sample, 0.47, cell_rng("s", "ego_drift", 3, 0))
        assert out.realized["outage_distance_m"] == pytest.approx(300.0)
        assert out.realized["pos_offset_m"] == pytest.approx(1.41, rel=1e-6)

    def test_error_is_largest_at_the_current_pose(self, sample):
        out = build("ego_drift").apply(sample, 2.0, cell_rng("s", "ego_drift", 7, 0))
        err = np.linalg.norm(
            out.sample.ego_history[:, [EGO_X, EGO_Y]] - sample.ego_history[:, [EGO_X, EGO_Y]],
            axis=1,
        )
        assert err[-1] > err[0]

    def test_heading_offset_follows_the_field_study_ratio(self, sample):
        """1.13 deg of heading RMS accompanying 0.47 %/distance."""
        p = build("ego_drift", heading_deg_per_pct=2.4, random_walk_frac=0.0)
        out = p.apply(sample, 0.47, cell_rng("s", "ego_drift", 3, 0))
        assert out.realized["yaw_offset_deg"] == pytest.approx(1.13, abs=0.02)


class TestStaleness:
    def test_lag_lands_on_the_right_waypoint(self, sample):
        """tau = 500 ms at a 100 ms period is a 5-waypoint shift, so the reported
        current pose is the pose from 5 waypoints ago."""
        out = build("staleness").apply(sample, 500.0, cell_rng("s", "staleness", 4, 0))
        assert out.realized["shift_waypoints"] == 5.0
        assert np.allclose(out.sample.ego_history[-1], sample.ego_history[-6])

    def test_lag_distance_matches_speed_times_tau(self, sample):
        """10 m/s for 500 ms is 5 m of lag."""
        out = build("staleness").apply(sample, 500.0, cell_rng("s", "staleness", 4, 0))
        assert out.realized["lag_m"] == pytest.approx(5.0)

    def test_is_deterministic_regardless_of_rng(self, sample):
        """Why configs/axis/staleness.yaml sets seeds: 1 -- a second seed is a
        bit-identical duplicate and would burn GPU-hours on a copy."""
        a = build("staleness").apply(sample, 700.0, cell_rng("s", "staleness", 6, 0))
        b = build("staleness").apply(sample, 700.0, cell_rng("s", "staleness", 6, 1))
        assert np.array_equal(a.sample.ego_history, b.sample.ego_history)

    def test_off_grid_tau_rounds_and_reports_realized_lag(self, sample):
        """PLAN.md 3.2 puts tau on multiples of the frame period; anything off the
        grid rounds to the nearest waypoint and logs what it actually applied."""
        requested = 235.0
        out = build("staleness").apply(sample, requested, cell_rng("s", "staleness", 2, 0))
        assert out.realized["shift_waypoints"] == 2.0
        assert out.realized["lag_ms"] == pytest.approx(200.0)
        assert out.realized["lag_ms"] != requested

    def test_extrapolates_past_the_history_window(self, sample):
        """tau = 2000 ms exceeds the 1.6 s window, so the whole history is
        extrapolated backwards rather than clamped -- clamping would flatten
        every level above 1.6 s into one and break monotonicity at the top of
        the grid."""
        p = build("staleness")
        far = p.apply(sample, 2000.0, cell_rng("s", "staleness", 8, 0))
        near = p.apply(sample, 1500.0, cell_rng("s", "staleness", 7, 0))
        assert far.realized["lag_m"] > near.realized["lag_m"]
        # Constant-velocity extrapolation preserves the straight-line path.
        assert far.sample.ego_history[:, EGO_YAW] == pytest.approx(0.0)
        spacing = np.diff(far.sample.ego_history[:, EGO_X])
        assert spacing == pytest.approx(spacing[0])
