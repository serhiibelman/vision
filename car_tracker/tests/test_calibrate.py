"""
Tests for measuring the pixel-to-ground transform.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from car_tracker.calibrate import (
    CalibrationError,
    circular_mean_deg,
    fit_yaw_convention,
    measure_shift,
    select_pairs,
    wrap180,
)

SIZE = (640, 480)  # width, height


@pytest.fixture
def textured():
    """
    High-contrast noise, so corner detection and optical flow have plenty to lock on.
    """
    rng = np.random.default_rng(7)
    small = rng.integers(0, 255, size=(60, 80, 3), dtype=np.uint8)
    return cv2.resize(small, SIZE, interpolation=cv2.INTER_NEAREST)


def shifted(image, dx, dy, angle_deg=0.0, scale=1.0):
    """
    Apply a known similarity transform, so the measurement can be checked against it.
    """
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle_deg, scale)
    matrix[:, 2] += (dx, dy)
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)


@pytest.fixture
def telemetry():
    """
    Sixty frames drifting north at a steady altitude and heading.
    """
    n = 60
    return pd.DataFrame(
        {
            "frame": np.arange(1, n + 1),
            "t_sec": np.arange(n) / 30.0,
            "lat": 48.2670 + np.arange(n) * 5e-5,
            "lon": np.full(n, 25.9145),
            "rel_alt": np.full(n, 100.0),
            "yaw": np.full(n, -65.8),
            "gsd": np.full(n, 0.08),
        }
    )


class TestWrap180:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0, 0), (180, -180), (179, 179), (190, -170), (-190, 170), (360, 0)],
    )
    def test_wraps(self, value, expected):
        assert wrap180(value) == pytest.approx(expected)

    def test_vectorised(self):
        assert list(wrap180(np.array([350.0, 10.0]))) == pytest.approx([-10.0, 10.0])


class TestCircularMean:
    def test_simple_mean(self):
        assert circular_mean_deg(np.array([10.0, 20.0])) == pytest.approx(15.0)

    def test_handles_wraparound(self):
        """
        The plain arithmetic mean of these is 0, which is 180 degrees wrong.
        """
        assert abs(circular_mean_deg(np.array([179.0, -179.0]))) == pytest.approx(180.0)


class TestMeasureShift:
    def test_recovers_pure_translation(self, textured):
        result = measure_shift(textured, shifted(textured, 20, -12))
        assert result.dx == pytest.approx(20, abs=0.5)
        assert result.dy == pytest.approx(-12, abs=0.5)
        assert result.magnitude_px == pytest.approx(np.hypot(20, 12), abs=0.5)

    def test_recovers_rotation(self, textured):
        result = measure_shift(textured, shifted(textured, 0, 0, angle_deg=5.0))
        # getRotationMatrix2D rotates counter-clockwise; image y grows downward, so
        # the fitted angle carries the opposite sign.
        assert result.rotation_deg == pytest.approx(-5.0, abs=0.6)

    def test_recovers_scale(self, textured):
        result = measure_shift(textured, shifted(textured, 0, 0, scale=1.1))
        assert result.scale == pytest.approx(1.1, abs=0.02)

    def test_identical_frames_give_zero_shift(self, textured):
        result = measure_shift(textured, textured.copy())
        assert result.magnitude_px == pytest.approx(0.0, abs=0.2)

    def test_reports_inlier_and_tracked_counts(self, textured):
        result = measure_shift(textured, shifted(textured, 10, 0))
        assert result.inliers > 30
        assert result.tracked >= result.inliers
        assert len(result.points_a) == result.inliers
        assert result.points_a.shape[1] == 2

    def test_moving_objects_are_rejected_as_outliers(self, textured):
        """
        Cars move against the dominant ground motion; RANSAC must discard them, so no
        masking is needed before fitting.
        """
        moved = shifted(textured, 15, 0)
        # Paste a bright block in a different place in each frame, mimicking a vehicle
        # travelling independently of the ground.
        before, after = textured.copy(), moved.copy()
        before[200:240, 100:180] = 255
        after[200:240, 300:380] = 255
        result = measure_shift(before, after)
        assert result.dx == pytest.approx(15, abs=1.0)

    def test_blank_frame_raises(self):
        blank = np.zeros((*SIZE[::-1], 3), dtype=np.uint8)
        with pytest.raises(CalibrationError, match="corners"):
            measure_shift(blank, blank)


class TestSelectPairs:
    def test_returns_expected_columns(self, telemetry):
        pairs = select_pairs(telemetry, gap=15, count=5, min_distance_m=1.0)
        assert {"frame_a", "frame_b", "gps_distance_m", "gps_bearing_deg", "gsd_spec"} <= set(
            pairs.columns
        )

    def test_gap_is_respected(self, telemetry):
        pairs = select_pairs(telemetry, gap=15, count=5, min_distance_m=1.0)
        assert ((pairs["frame_b"] - pairs["frame_a"]) == 15).all()

    def test_honours_count(self, telemetry):
        assert len(select_pairs(telemetry, gap=10, count=3, min_distance_m=1.0)) <= 3

    def test_rejects_pairs_that_move_too_little(self, telemetry):
        with pytest.raises(CalibrationError, match="no frame pair"):
            select_pairs(telemetry, gap=2, min_distance_m=500.0)

    def test_rejects_pairs_that_rotate(self, telemetry):
        telemetry["yaw"] = np.linspace(0, 180, len(telemetry))
        with pytest.raises(CalibrationError, match="no frame pair"):
            select_pairs(telemetry, gap=15, min_distance_m=1.0, max_yaw_delta_deg=0.01)

    def test_rejects_pairs_that_change_altitude(self, telemetry):
        telemetry["rel_alt"] = np.linspace(100, 60, len(telemetry))
        with pytest.raises(CalibrationError, match="no frame pair"):
            select_pairs(telemetry, gap=15, min_distance_m=1.0, max_alt_delta_m=0.01)

    def test_bearing_is_north_for_northward_flight(self, telemetry):
        pairs = select_pairs(telemetry, gap=15, count=3, min_distance_m=1.0)
        assert pairs["gps_bearing_deg"].iloc[0] == pytest.approx(0.0, abs=1.0)


class TestFitYawConvention:
    @pytest.mark.parametrize("sign", [1, -1])
    @pytest.mark.parametrize("offset", [0.0, 12.0, -30.0])
    def test_recovers_synthetic_convention(self, sign, offset):
        yaw = np.array([-170.0, -90.0, -20.0, 35.0, 120.0, 175.0])
        measurements = pd.DataFrame(
            {"yaw": yaw, "rotation_deg": wrap180(sign * yaw + offset)}
        )
        got_sign, got_offset, spread = fit_yaw_convention(measurements)
        assert got_sign == sign
        assert wrap180(got_offset - offset) == pytest.approx(0.0, abs=0.1)
        assert spread == pytest.approx(0.0, abs=0.1)

    def test_spread_reports_inconsistency(self):
        """
        Random rotations fit neither sign, and the spread must say so.
        """
        rng = np.random.default_rng(3)
        measurements = pd.DataFrame(
            {
                "yaw": np.linspace(-170, 170, 20),
                "rotation_deg": rng.uniform(-180, 180, 20),
            }
        )
        _, _, spread = fit_yaw_convention(measurements)
        assert spread > 30.0
