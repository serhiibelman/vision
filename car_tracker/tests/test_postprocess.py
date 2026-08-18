
"""
Tests for separating moving vehicles from parked ones.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from car_tracker.geo import metres_to_latlon
from car_tracker.postprocess import (
    MovingCriteria,
    classify,
    estimate_gps_drift,
    moving_tracks,
    remove_gps_drift,
    report,
    smooth_positions,
    to_geojson,
    track_features,
)

LAT, LON = 48.267013, 25.914562
FPS = 30.0


def build(paths, conf=0.8):
    """
    Assemble a tracks table from per-track ENU paths.

    Args:
        paths: mapping of track_id -> (n, 2) array of east/north metres.
    """
    rows = []
    for track_id, path in paths.items():
        previous = None
        for index, (east, north) in enumerate(path):
            lat, lon = metres_to_latlon(LAT, LON, east, north)
            step = 0.0 if previous is None else np.hypot(east - previous[0], north - previous[1])
            speed = float(step * FPS)
            rows.append(
                {
                    "track_id": track_id,
                    "frame": index + 1,
                    "t_sec": index / FPS,
                    "lat": float(lat),
                    "lon": float(lon),
                    "east_m": float(east),
                    "north_m": float(north),
                    "speed_mps": speed,
                    "heading_deg": 90.0,
                    "conf": conf,
                }
            )
            previous = (east, north)
    return pd.DataFrame(rows)


def driving(east0=0.0, north0=0.0, speed=12.0, n=60):
    """
    A car covering ground at a steady speed: 12 m/s for 2 s is ~24 m.
    """
    t = np.arange(n) / FPS
    return np.stack([east0 + speed * t, np.full(n, north0)], axis=1)


def parked(east=0.0, north=0.0, n=60, jitter=1.0, seed=0):
    """
    A stationary car whose apparent position wanders with noise.
    """
    rng = np.random.default_rng(seed)
    return np.stack([np.full(n, east), np.full(n, north)], axis=1) + rng.normal(
        0.0, jitter, size=(n, 2)
    )


@pytest.fixture
def mixed():
    """
    One car driving past five parked ones — the shape of the real footage.
    """
    paths = {1: driving(n=90, speed=14.0)}
    for index in range(2, 7):
        paths[index] = parked(east=index * 12.0, north=25.0, n=90, seed=index)
    return build(paths)


class TestTrackFeatures:
    def test_empty_input(self):
        assert track_features(pd.DataFrame()).empty

    def test_driving_car_has_displacement_and_straightness(self):
        features = track_features(build({1: driving(speed=12.0, n=60)}))
        row = features.iloc[0]
        assert row["displacement_m"] == pytest.approx(12.0 * 59 / FPS, rel=0.05)
        assert row["straightness"] == pytest.approx(1.0, abs=0.02)

    def test_parked_car_wanders_without_displacing(self):
        row = track_features(build({1: parked(jitter=1.0, n=60)})).iloc[0]
        assert row["displacement_m"] < 5.0
        assert row["path_length_m"] > 20.0
        assert row["straightness"] < 0.2

    def test_zero_length_path_is_not_straight(self):
        """
        Guards a divide-by-zero: a perfectly still car must not look perfectly straight.
        """
        row = track_features(build({1: parked(jitter=0.0, n=10)})).iloc[0]
        assert row["path_length_m"] == pytest.approx(0.0)
        assert row["straightness"] == 0.0

    def test_records_endpoints_and_counts(self):
        features = track_features(build({1: driving(n=30)}))
        row = features.iloc[0]
        assert row["observations"] == 30
        assert row["duration_s"] == pytest.approx(29 / FPS)
        assert row["start_lat"] != row["end_lat"] or row["start_lon"] != row["end_lon"]

    def test_one_row_per_track(self, mixed):
        assert len(track_features(mixed)) == 6


class TestClassify:
    def test_driving_car_is_moving(self):
        features = classify(track_features(build({1: driving(speed=14.0, n=90)})))
        assert features["is_moving"].all()
        assert features["verdict"].iloc[0] == "moving"

    def test_parked_car_is_rejected(self):
        features = classify(track_features(build({1: parked(jitter=1.0, n=90)})))
        assert not features["is_moving"].any()
        assert features["verdict"].iloc[0] in {"stationary", "wandering", "too_slow"}

    def test_short_track_is_rejected_as_too_short(self):
        features = classify(track_features(build({1: driving(n=3)})))
        assert features["verdict"].iloc[0] == "too_short"

    def test_implausible_speed_is_rejected(self):
        features = track_features(build({1: driving(speed=14.0, n=90)}))
        features.loc[0, "max_speed_mps"] = 200.0
        assert classify(features)["verdict"].iloc[0] == "implausible_speed"

    def test_slow_creep_is_rejected_as_stationary(self):
        """
        A car nudging forward 2 m over 3 s is not a journey worth mapping.
        """
        features = classify(track_features(build({1: driving(speed=0.7, n=90)})))
        assert not features["is_moving"].any()

    def test_verdict_names_the_failed_test(self):
        features = classify(track_features(build({1: parked(jitter=0.0, n=90)})))
        assert features["verdict"].iloc[0] == "stationary"

    def test_thresholds_are_configurable(self):
        tracks = build({1: driving(speed=3.0, n=30)})
        strict = classify(track_features(tracks), MovingCriteria(min_displacement_m=100.0))
        lenient = classify(
            track_features(tracks),
            MovingCriteria(min_displacement_m=1.0, min_average_speed_mps=0.1, min_observations=2),
        )
        assert not strict["is_moving"].any()
        assert lenient["is_moving"].all()

    def test_empty_input(self):
        assert classify(track_features(pd.DataFrame())).empty


class TestGpsDrift:
    def test_no_drift_when_everything_is_still(self):
        drift = estimate_gps_drift(build({1: parked(jitter=0.0, n=20)}))
        assert drift["drift_east_m"].abs().max() == pytest.approx(0.0, abs=1e-9)

    def test_detects_common_mode_drift(self):
        """
        Every parked car shifting in formation is the drone's fix moving, not traffic.
        """
        n = 40
        ramp = np.stack([np.linspace(0, 8, n), np.zeros(n)], axis=1)
        paths = {i: parked(east=i * 15.0, n=n, jitter=0.0) + ramp for i in range(1, 5)}
        drift = estimate_gps_drift(build(paths))
        assert drift["drift_east_m"].iloc[-1] == pytest.approx(8.0, rel=0.05)

    def test_correction_removes_apparent_motion(self):
        n = 40
        ramp = np.stack([np.linspace(0, 8, n), np.zeros(n)], axis=1)
        paths = {i: parked(east=i * 15.0, n=n, jitter=0.0) + ramp for i in range(1, 5)}
        corrected = remove_gps_drift(build(paths))
        features = track_features(corrected, "east_corrected_m", "north_corrected_m")
        assert features["displacement_m"].max() < 1.0

    def test_a_real_mover_survives_the_correction(self, mixed):
        """
        The correction must not cancel the one car that is genuinely driving.
        """
        corrected = remove_gps_drift(mixed)
        features = track_features(corrected, "east_corrected_m", "north_corrected_m")
        assert features.loc[features["track_id"] == 1, "displacement_m"].iloc[0] > 20.0

    def test_adds_columns_on_empty_input(self):
        empty = pd.DataFrame(columns=["track_id", "t_sec", "east_m", "north_m"])
        assert "east_corrected_m" in remove_gps_drift(empty).columns


class TestSmoothing:
    def test_rejects_a_single_outlier(self):
        """
        A median window drops a bad detection outright; a mean would blend it in.
        """
        path = driving(speed=10.0, n=40)
        path[20] += 50.0
        smoothed = smooth_positions(build({1: path}))
        assert abs(smoothed["east_smooth_m"].iloc[20] - path[20, 0]) > 20.0

    def test_preserves_a_clean_path(self):
        path = driving(speed=10.0, n=40)
        smoothed = smooth_positions(build({1: path}))
        assert smoothed["east_smooth_m"].iloc[20] == pytest.approx(path[20, 0], abs=0.5)

    def test_keeps_tracks_separate(self, mixed):
        smoothed = smooth_positions(mixed)
        assert smoothed["track_id"].nunique() == 6
        assert len(smoothed) == len(mixed)

    def test_empty_input(self):
        assert smooth_positions(pd.DataFrame()).empty


class TestMovingTracks:
    def test_keeps_the_mover_and_drops_the_parked(self, mixed):
        observations, features = moving_tracks(mixed)
        assert set(observations["track_id"]) == {1}
        assert int(features["is_moving"].sum()) == 1

    def test_every_track_is_reported_with_a_verdict(self, mixed):
        _, features = moving_tracks(mixed)
        assert len(features) == 6
        assert features["verdict"].notna().all()

    def test_all_parked_yields_nothing(self):
        paths = {i: parked(east=i * 12.0, n=60, seed=i) for i in range(1, 5)}
        observations, features = moving_tracks(build(paths))
        assert observations.empty
        assert not features["is_moving"].any()

    def test_empty_input(self):
        observations, features = moving_tracks(pd.DataFrame())
        assert observations.empty
        assert features.empty


class TestGeoJson:
    def test_writes_a_feature_collection(self, mixed, tmp_path):
        observations, features = moving_tracks(mixed)
        path = to_geojson(observations, features, tmp_path / "sub" / "tracks.geojson")
        data = json.loads(path.read_text())
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 1

    def test_geometry_is_a_linestring_in_lon_lat_order(self, mixed, tmp_path):
        observations, features = moving_tracks(mixed)
        data = json.loads(to_geojson(observations, features, tmp_path / "t.geojson").read_text())
        geometry = data["features"][0]["geometry"]
        assert geometry["type"] == "LineString"
        first_lon, first_lat = geometry["coordinates"][0]
        assert 25.0 < first_lon < 26.0
        assert 48.0 < first_lat < 49.0

    def test_properties_carry_measurements(self, mixed, tmp_path):
        observations, features = moving_tracks(mixed)
        data = json.loads(to_geojson(observations, features, tmp_path / "t.geojson").read_text())
        properties = data["features"][0]["properties"]
        assert properties["track_id"] == 1
        assert properties["displacement_m"] > 20
        assert properties["median_speed_kmh"] == pytest.approx(
            properties["median_speed_mps"] * 3.6, rel=0.02
        )

    def test_empty_collection_is_still_valid(self, tmp_path):
        path = to_geojson(pd.DataFrame(columns=["track_id", "t_sec", "lat", "lon"]),
                          pd.DataFrame(columns=["track_id"]), tmp_path / "e.geojson")
        assert json.loads(path.read_text())["features"] == []


class TestReport:
    def test_no_tracks(self):
        assert report(pd.DataFrame()) == "no tracks"

    def test_counts_moving_and_rejections(self, mixed):
        _, features = moving_tracks(mixed)
        text = report(features)
        assert "moving            1" in text
        assert "rejected" in text

    def test_includes_speed_summary(self, mixed):
        _, features = moving_tracks(mixed)
        assert "median speed" in report(features)


class TestAverageSpeedCriterion:
    """
    Motion is judged by displacement / duration, not by the Kalman filter's own speed.

    The filter starts at rest, so its median speed understates by 1.7x typically and up to
    49x on short tracks — enough to reject real cars doing 60 km/h as "too slow".
    """

    def test_average_speed_is_displacement_over_duration(self):
        features = track_features(build({1: driving(speed=12.0, n=60)}))
        row = features.iloc[0]
        assert row["average_speed_mps"] == pytest.approx(
            row["displacement_m"] / row["duration_s"], rel=1e-6
        )

    def test_real_car_survives_a_broken_filtered_speed(self):
        """
        The exact regression: 39 m in 2.3 s, but the filter reports 1.1 m/s.
        """
        features = track_features(build({1: driving(speed=17.0, n=70)}))
        features.loc[0, "median_speed_mps"] = 1.1      # what the filter actually said
        verdict = classify(features)
        assert verdict["is_moving"].all()

    def test_slow_creep_is_still_rejected(self):
        features = classify(track_features(build({1: driving(speed=1.0, n=90)})))
        assert not features["is_moving"].any()

    def test_zero_duration_does_not_divide_by_zero(self):
        one = build({1: driving(n=1)})
        assert track_features(one)["average_speed_mps"].iloc[0] == 0.0


class TestDriftCorrectionIsOptional:
    """
    Drift correction is off by default: its estimator sums a per-frame median, and the
    small bias in that median integrates into unbounded false drift (117 m on real data).
    """

    def test_off_by_default(self, mixed):
        observations, features = moving_tracks(mixed)
        assert "east_corrected_m" not in observations.columns

    def test_correction_actually_reaches_the_measurement(self):
        """
        It used to be computed and then discarded: smoothing always read the raw columns,
        so enabling the flag changed nothing at all.
        """
        paths = {100 + i: driving(east0=i * 25.0, speed=12.0, n=90) for i in range(6)}
        paths.update({200 + i: parked(east=i * 12.0, north=30.0, n=90, seed=i) for i in range(6)})
        tracks = build(paths)
        _, off = moving_tracks(tracks, correct_drift=False)
        _, on = moving_tracks(tracks, correct_drift=True)
        assert int(off["is_moving"].sum()) != int(on["is_moving"].sum())

    def test_can_be_enabled(self, mixed):
        observations, _ = moving_tracks(mixed, correct_drift=True)
        assert "east_corrected_m" in observations.columns

    def test_traffic_moving_together_survives_by_default(self):
        """
        With correction on, several vehicles moving together look like drone drift and get
        cancelled. This is what made real movers disappear in dense traffic.
        """
        paths = {100 + i: driving(east0=i * 25.0, speed=12.0, n=90) for i in range(6)}
        paths.update({200 + i: parked(east=i * 12.0, north=30.0, n=90, seed=i) for i in range(6)})
        tracks = build(paths)

        _, without = moving_tracks(tracks, correct_drift=False)
        _, with_correction = moving_tracks(tracks, correct_drift=True)
        assert int(without["is_moving"].sum()) == 6
        assert int(with_correction["is_moving"].sum()) < 6
