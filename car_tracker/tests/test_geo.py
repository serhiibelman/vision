"""
Tests for pixel-to-ground projection.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from car_tracker.geo import (
    Calibration,
    CameraModel,
    bearing_deg,
    haversine_m,
    latlon_to_metres,
    metres_to_latlon,
)

LAT, LON = 48.267013, 25.914562
GSD = 0.08


@pytest.fixture
def model():
    return CameraModel()


@pytest.fixture
def telemetry():
    n = 3
    return pd.DataFrame(
        {
            "frame": [1, 2, 3],
            "t_sec": [0.0, 0.033, 0.066],
            "lat": [LAT] * n,
            "lon": [LON] * n,
            "rel_alt": [102.0] * n,
            "gsd": [GSD] * n,
            "yaw": [0.0, 90.0, 180.0],
        }
    )


class TestMetreConversions:
    def test_north_offset_increases_latitude(self):
        lat, lon = metres_to_latlon(LAT, LON, east_m=0, north_m=111.2)
        assert lat > LAT
        assert lat == pytest.approx(LAT + 0.001, abs=1e-5)
        assert lon == pytest.approx(LON)

    def test_east_offset_increases_longitude(self):
        lat, lon = metres_to_latlon(LAT, LON, east_m=100.0, north_m=0)
        assert lon > LON
        assert lat == pytest.approx(LAT)

    def test_round_trip(self):
        east, north = 137.0, -412.0
        lat, lon = metres_to_latlon(LAT, LON, east, north)
        back_e, back_n = latlon_to_metres(LAT, LON, lat, lon)
        assert back_e == pytest.approx(east, abs=0.01)
        assert back_n == pytest.approx(north, abs=0.01)

    def test_longitude_scale_depends_on_latitude(self):
        """
        A metre east is more degrees of longitude near the poles than at the equator.
        """
        _, lon_equator = metres_to_latlon(0.0, 0.0, east_m=1000.0, north_m=0)
        _, lon_high = metres_to_latlon(60.0, 0.0, east_m=1000.0, north_m=0)
        assert abs(lon_high) > abs(lon_equator)

    def test_broadcasts_over_arrays(self):
        lat, lon = metres_to_latlon(
            np.array([LAT, LAT]), np.array([LON, LON]),
            np.array([0.0, 100.0]), np.array([100.0, 0.0]),
        )
        assert lat.shape == (2,)
        assert lat[0] > lat[1]


class TestDistanceAndBearing:
    def test_known_distance(self):
        assert haversine_m(48.0, 25.0, 49.0, 25.0) == pytest.approx(111_200, rel=0.01)

    def test_zero_distance(self):
        assert haversine_m(LAT, LON, LAT, LON) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize(
        ("dlat", "dlon", "expected"),
        [(0.001, 0.0, 0.0), (0.0, 0.001, 90.0), (-0.001, 0.0, 180.0), (0.0, -0.001, 270.0)],
    )
    def test_cardinal_bearings(self, dlat, dlon, expected):
        got = bearing_deg(LAT, LON, LAT + dlat, LON + dlon)
        assert got == pytest.approx(expected, abs=0.5)

    def test_bearing_is_in_range(self):
        got = bearing_deg(LAT, LON, LAT - 0.01, LON - 0.01)
        assert 0.0 <= got < 360.0


class TestCalibration:
    def test_defaults_are_identity(self):
        cal = Calibration()
        assert cal.gsd_scale == 1.0
        assert cal.yaw_offset_deg == 0.0
        assert cal.yaw_sign == 1
        assert cal.residual_m is None

    @pytest.mark.parametrize("sign", [0, 2, -2])
    def test_rejects_invalid_sign(self, sign):
        with pytest.raises(ValueError, match="yaw_sign"):
            Calibration(yaw_sign=sign)

    @pytest.mark.parametrize("scale", [0.0, -1.0])
    def test_rejects_non_positive_scale(self, scale):
        with pytest.raises(ValueError, match="gsd_scale"):
            Calibration(gsd_scale=scale)

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            Calibration().gsd_scale = 2.0


class TestCentre:
    def test_matches_image_size(self, model):
        assert model.centre == (959.5, 539.5)

    def test_custom_size(self):
        assert CameraModel(image_size=(101, 51)).centre == (50.0, 25.0)


class TestPixelToLatLon:
    def test_centre_pixel_is_the_drone_position(self, model):
        cx, cy = model.centre
        lat, lon = model.pixel_to_latlon(cx, cy, LAT, LON, GSD, yaw_deg=0.0)
        assert lat == pytest.approx(LAT, abs=1e-9)
        assert lon == pytest.approx(LON, abs=1e-9)

    def test_centre_is_invariant_to_yaw(self, model):
        """
        Rotation is about the image centre, so the centre never moves.
        """
        cx, cy = model.centre
        for yaw in (0.0, 45.0, -120.0, 180.0):
            lat, lon = model.pixel_to_latlon(cx, cy, LAT, LON, GSD, yaw)
            assert lat == pytest.approx(LAT, abs=1e-9)
            assert lon == pytest.approx(LON, abs=1e-9)

    def test_yaw_zero_maps_image_up_to_north(self, model):
        cx, cy = model.centre
        lat, lon = model.pixel_to_latlon(cx, cy - 100, LAT, LON, GSD, yaw_deg=0.0)
        assert lat > LAT
        assert lon == pytest.approx(LON, abs=1e-9)

    def test_yaw_ninety_maps_image_up_to_east(self, model):
        cx, cy = model.centre
        lat, lon = model.pixel_to_latlon(cx, cy - 100, LAT, LON, GSD, yaw_deg=90.0)
        assert lon > LON
        assert lat == pytest.approx(LAT, abs=1e-9)

    def test_yaw_one_eighty_maps_image_up_to_south(self, model):
        cx, cy = model.centre
        lat, _ = model.pixel_to_latlon(cx, cy - 100, LAT, LON, GSD, yaw_deg=180.0)
        assert lat < LAT

    def test_offset_distance_equals_pixels_times_gsd(self, model):
        """
        100 px at 8 cm/px must land 8 m away, whatever the heading.
        """
        cx, cy = model.centre
        for yaw in (0.0, 37.0, -145.0):
            lat, lon = model.pixel_to_latlon(cx, cy - 100, LAT, LON, GSD, yaw)
            assert haversine_m(LAT, LON, lat, lon) == pytest.approx(8.0, rel=1e-3)

    def test_gsd_scale_scales_the_offset(self):
        cx, cy = CameraModel().centre
        plain = CameraModel()
        scaled = CameraModel(calibration=Calibration(gsd_scale=2.0))
        d_plain = haversine_m(
            LAT, LON, *plain.pixel_to_latlon(cx, cy - 100, LAT, LON, GSD, 0.0)
        )
        d_scaled = haversine_m(
            LAT, LON, *scaled.pixel_to_latlon(cx, cy - 100, LAT, LON, GSD, 0.0)
        )
        assert d_scaled == pytest.approx(2 * d_plain, rel=1e-6)

    def test_yaw_sign_flip_mirrors_the_result(self):
        """
        The failure mode the calibration gate exists to catch.
        """
        cx, cy = CameraModel().centre
        normal = CameraModel()
        flipped = CameraModel(calibration=Calibration(yaw_sign=-1))
        _, lon_n = normal.pixel_to_latlon(cx, cy - 100, LAT, LON, GSD, 90.0)
        _, lon_f = flipped.pixel_to_latlon(cx, cy - 100, LAT, LON, GSD, 90.0)
        assert (lon_n - LON) == pytest.approx(-(lon_f - LON), rel=1e-6)

    def test_yaw_offset_rotates_the_result(self):
        cx, cy = CameraModel().centre
        offset = CameraModel(calibration=Calibration(yaw_offset_deg=90.0))
        lat, lon = offset.pixel_to_latlon(cx, cy - 100, LAT, LON, GSD, yaw_deg=0.0)
        assert lon > LON
        assert lat == pytest.approx(LAT, abs=1e-9)

    def test_broadcasts_over_arrays(self, model):
        cx, cy = model.centre
        lat, lon = model.pixel_to_latlon(
            np.array([cx, cx]), np.array([cy - 100, cy + 100]),
            LAT, LON, GSD, yaw_deg=0.0,
        )
        assert lat[0] > LAT > lat[1]


class TestRotationDeg:
    def test_identity_by_default(self, model):
        assert model.rotation_deg(42.0) == pytest.approx(42.0)

    def test_applies_sign_then_offset(self):
        model = CameraModel(calibration=Calibration(yaw_sign=-1, yaw_offset_deg=10.0))
        assert model.rotation_deg(30.0) == pytest.approx(-20.0)


class TestProjectDetections:
    def test_adds_lat_lon(self, model, telemetry):
        cx, cy = model.centre
        det = pd.DataFrame({"frame": [1, 2], "u": [cx, cx], "v": [cy, cy]})
        out = model.project_detections(det, telemetry)
        assert {"lat", "lon"}.issubset(out.columns)
        assert out["lat"].iloc[0] == pytest.approx(LAT, abs=1e-9)

    def test_row_count_preserved(self, model, telemetry):
        det = pd.DataFrame({"frame": [1, 1, 2], "u": [10.0, 20.0, 30.0], "v": [1.0, 2.0, 3.0]})
        assert len(model.project_detections(det, telemetry)) == 3

    def test_uses_per_frame_yaw(self, model, telemetry):
        """
        Frames 1 and 2 differ only in yaw, so the same pixel must project differently.
        """
        cx, cy = model.centre
        det = pd.DataFrame({"frame": [1, 2], "u": [cx, cx], "v": [cy - 100, cy - 100]})
        out = model.project_detections(det, telemetry)
        assert out["lat"].iloc[0] > out["lat"].iloc[1] or out["lon"].iloc[1] > LON

    def test_missing_column_raises(self, model, telemetry):
        with pytest.raises(KeyError, match="'v'"):
            model.project_detections(pd.DataFrame({"frame": [1], "u": [0.0]}), telemetry)

    def test_unknown_frame_raises(self, model, telemetry):
        det = pd.DataFrame({"frame": [99], "u": [0.0], "v": [0.0]})
        with pytest.raises(ValueError, match="no telemetry for frames"):
            model.project_detections(det, telemetry)

    def test_custom_pixel_columns(self, model, telemetry):
        cx, cy = model.centre
        det = pd.DataFrame({"frame": [1], "cx": [cx], "cy": [cy]})
        out = model.project_detections(det, telemetry, u_col="cx", v_col="cy")
        assert out["lat"].iloc[0] == pytest.approx(LAT, abs=1e-9)


class TestFootprint:
    def test_matches_gsd_times_pixels(self, model):
        width_m, height_m = model.footprint_m(GSD)
        assert width_m == pytest.approx(1920 * GSD)
        assert height_m == pytest.approx(1080 * GSD)

    def test_respects_gsd_scale(self):
        model = CameraModel(calibration=Calibration(gsd_scale=1.3))
        assert model.footprint_m(GSD)[0] == pytest.approx(1920 * GSD * 1.3)


def test_repr_shows_calibration():
    model = CameraModel(calibration=Calibration(gsd_scale=1.27, yaw_sign=-1))
    assert "1.270" in repr(model)
    assert "-1*yaw" in repr(model)
