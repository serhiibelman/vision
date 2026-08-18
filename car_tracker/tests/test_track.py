
"""
Tests for linking detections into tracks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from car_tracker.geo import metres_to_latlon
from car_tracker.track import (
    COLUMNS,
    KalmanTrack,
    TrackingError,
    summarise,
    to_local_metres,
    track_detections,
)

LAT, LON = 48.267013, 25.914562
FPS = 30.0


def make_detections(paths, frames, conf=0.8):
    """
    Build a projected-detection table from paths given in ENU metres.

    Args:
        paths: list of (east, north) arrays, one per vehicle, each ``frames`` long.
        frames: number of frames.
    """
    rows = []
    for index in range(frames):
        for path in paths:
            east, north = path[index]
            if not np.isfinite(east):
                continue
            lat, lon = metres_to_latlon(LAT, LON, east, north)
            rows.append(
                {
                    "frame": index + 1,
                    "t_sec": index / FPS,
                    "lat": float(lat),
                    "lon": float(lon),
                    "conf": conf,
                }
            )
    return pd.DataFrame(rows)


def straight(east0, north0, v_east, v_north, frames):
    """
    A vehicle travelling at constant velocity.
    """
    t = np.arange(frames) / FPS
    return np.stack([east0 + v_east * t, north0 + v_north * t], axis=1)


def stationary(east, north, frames, jitter=0.0, seed=0):
    """
    A parked car, optionally with GPS-like noise.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, jitter, size=(frames, 2)) if jitter else np.zeros((frames, 2))
    return np.stack([np.full(frames, east), np.full(frames, north)], axis=1) + noise


class TestKalmanTrack:
    def test_starts_at_measurement_with_zero_velocity(self):
        filter_ = KalmanTrack(10.0, 20.0, 0.0)
        assert tuple(filter_.position) == (10.0, 20.0)
        assert filter_.speed_mps == pytest.approx(0.0)

    def test_predict_moves_along_velocity(self):
        filter_ = KalmanTrack(0.0, 0.0, 0.0)
        for step in range(1, 30):
            filter_.predict(1 / FPS)
            filter_.update(step * 0.5, 0.0)  # 15 m/s east
        assert filter_.velocity[0] == pytest.approx(15.0, rel=0.15)
        assert filter_.velocity[1] == pytest.approx(0.0, abs=1.0)

    def test_converges_on_a_steady_position(self):
        filter_ = KalmanTrack(0.0, 0.0, 0.0)
        for _ in range(40):
            filter_.predict(1 / FPS)
            filter_.update(5.0, -3.0)
        assert filter_.position == pytest.approx([5.0, -3.0], abs=0.5)
        assert filter_.speed_mps == pytest.approx(0.0, abs=1.0)

    def test_zero_dt_does_not_move(self):
        filter_ = KalmanTrack(4.0, 5.0, 0.0)
        assert tuple(filter_.predict(0.0)) == (4.0, 5.0)

    @pytest.mark.parametrize(
        ("v_east", "v_north", "expected"),
        [(1, 0, 90.0), (0, 1, 0.0), (-1, 0, 270.0), (0, -1, 180.0)],
    )
    def test_heading_is_clockwise_from_north(self, v_east, v_north, expected):
        filter_ = KalmanTrack(0.0, 0.0, 0.0)
        filter_.state[2:] = (v_east, v_north)
        assert filter_.heading_deg == pytest.approx(expected)

    def test_smooths_noisy_measurements(self):
        """
        The filtered position must sit closer to truth than the raw measurements do.
        """
        rng = np.random.default_rng(1)
        filter_ = KalmanTrack(0.0, 0.0, 0.0)
        errors = []
        for _ in range(50):
            filter_.predict(1 / FPS)
            noisy = rng.normal(0.0, 2.0, 2)
            filter_.update(*noisy)
            errors.append(np.hypot(*filter_.position))
        assert np.mean(errors[10:]) < 2.0


class TestToLocalMetres:
    def test_first_detection_is_the_origin(self):
        detections = make_detections([straight(0, 0, 0, 0, 3)], frames=3)
        localised, origin = to_local_metres(detections)
        assert origin == pytest.approx((LAT, LON))
        assert localised[["east_m", "north_m"]].iloc[0].tolist() == pytest.approx([0, 0], abs=0.01)

    def test_round_trips_metres(self):
        detections = make_detections([straight(30, -40, 0, 0, 2)], frames=2)
        localised, _ = to_local_metres(detections, origin=(LAT, LON))
        assert localised["east_m"].iloc[0] == pytest.approx(30.0, abs=0.05)
        assert localised["north_m"].iloc[0] == pytest.approx(-40.0, abs=0.05)

    def test_explicit_origin_is_respected(self):
        detections = make_detections([straight(0, 0, 0, 0, 2)], frames=2)
        localised, origin = to_local_metres(detections, origin=(48.0, 25.0))
        assert origin == (48.0, 25.0)
        assert abs(localised["north_m"].iloc[0]) > 1000

    def test_missing_latlon_raises(self):
        with pytest.raises(KeyError, match="lat"):
            to_local_metres(pd.DataFrame({"frame": [1]}))

    def test_empty_raises(self):
        with pytest.raises(TrackingError, match="no detections"):
            to_local_metres(pd.DataFrame({"lat": [], "lon": []}))


class TestTrackDetections:
    def test_single_moving_car_gets_one_track(self):
        detections = make_detections([straight(0, 0, 10, 0, 20)], frames=20)
        tracks = track_detections(detections)
        assert tracks["track_id"].nunique() == 1
        assert len(tracks) == 20

    def test_columns_and_order(self):
        detections = make_detections([straight(0, 0, 10, 0, 10)], frames=10)
        tracks = track_detections(detections)
        assert list(tracks.columns) == list(COLUMNS)
        assert tracks["frame"].is_monotonic_increasing

    def test_two_separated_cars_get_two_tracks(self):
        detections = make_detections(
            [straight(0, 0, 10, 0, 20), straight(0, 200, 10, 0, 20)], frames=20
        )
        assert track_detections(detections)["track_id"].nunique() == 2

    def test_crossing_cars_keep_distinct_ids(self):
        """
        Two vehicles passing in opposite directions must not swap identities. Greedy
        nearest-neighbour matching fails this; Hungarian assignment does not.
        """
        frames = 30
        tracks = track_detections(
            make_detections(
                [straight(-40, 0, 12, 0, frames), straight(40, 4, -12, 0, frames)], frames=frames
            )
        )
        assert tracks["track_id"].nunique() == 2
        for _, group in tracks.groupby("track_id"):
            east = group["east_m"].to_numpy()
            # Each vehicle keeps travelling the same way for its whole life.
            assert np.all(np.diff(east) > 0) or np.all(np.diff(east) < 0)

    def test_speed_is_recovered(self):
        tracks = track_detections(make_detections([straight(0, 0, 15, 0, 40)], frames=40))
        assert tracks["speed_mps"].iloc[-1] == pytest.approx(15.0, rel=0.15)

    def test_heading_is_recovered(self):
        tracks = track_detections(make_detections([straight(0, 0, 0, 12, 40)], frames=40))
        assert tracks["heading_deg"].iloc[-1] == pytest.approx(0.0, abs=10.0)

    def test_parked_car_is_tracked_with_near_zero_speed(self):
        """
        Parked cars are kept deliberately; step 6 removes them.
        """
        tracks = track_detections(
            make_detections([stationary(0, 0, 30, jitter=0.5)], frames=30)
        )
        assert tracks["track_id"].nunique() == 1
        assert tracks["speed_mps"].median() < 3.0

    def test_short_track_is_discarded(self):
        detections = make_detections([straight(0, 0, 10, 0, 2)], frames=2)
        assert track_detections(detections).empty

    def test_min_hits_is_configurable(self):
        detections = make_detections([straight(0, 0, 10, 0, 2)], frames=2)
        assert not track_detections(detections, min_hits=1).empty

    def test_gap_is_bridged_by_coasting(self):
        """
        A car missed for a few frames must keep its identity, not restart.
        """
        path = straight(0, 0, 10, 0, 30)
        path[10:14] = np.nan  # detector misses four frames
        tracks = track_detections(make_detections([path], frames=30))
        assert tracks["track_id"].nunique() == 1

    def test_long_gap_starts_a_new_track(self):
        path = straight(0, 0, 10, 0, 80)
        path[10:60] = np.nan  # missing for ~1.7 s, beyond max_coast_s
        tracks = track_detections(make_detections([path], frames=80))
        assert tracks["track_id"].nunique() == 2

    def test_teleporting_detection_is_not_associated(self):
        """
        Beyond max_speed * dt, a detection cannot be the same vehicle.
        """
        near = stationary(0, 0, 10)
        far = stationary(500, 500, 10)
        detections = pd.concat(
            [make_detections([near], frames=10), make_detections([far], frames=10)]
        )
        assert track_detections(detections)["track_id"].nunique() == 2

    def test_conf_is_carried_through(self):
        detections = make_detections([straight(0, 0, 10, 0, 10)], frames=10, conf=0.42)
        assert track_detections(detections)["conf"].iloc[0] == pytest.approx(0.42)

    def test_works_without_conf_column(self):
        detections = make_detections([straight(0, 0, 10, 0, 10)], frames=10).drop(columns="conf")
        tracks = track_detections(detections)
        assert tracks["conf"].isna().all()

    def test_missing_frame_column_raises(self):
        detections = make_detections([straight(0, 0, 1, 0, 5)], frames=5).drop(columns="frame")
        with pytest.raises(KeyError, match="frame"):
            track_detections(detections)

    def test_latlon_round_trips_back_to_input(self):
        detections = make_detections([straight(10, 20, 5, 0, 15)], frames=15)
        tracks = track_detections(detections)
        assert tracks["lat"].iloc[-1] == pytest.approx(detections["lat"].iloc[-1], abs=1e-5)


class TestSummarise:
    def test_empty_input(self):
        assert summarise(pd.DataFrame(columns=list(COLUMNS))).empty

    def test_moving_car_has_large_displacement(self):
        tracks = track_detections(make_detections([straight(0, 0, 10, 0, 30)], frames=30))
        row = summarise(tracks).iloc[0]
        assert row["displacement_m"] == pytest.approx(10.0 * 29 / FPS, rel=0.2)
        assert row["duration_s"] == pytest.approx(29 / FPS, rel=0.01)

    def test_parked_car_has_small_displacement_despite_path_length(self):
        """
        The distinction step 6 relies on: noise accumulates path length but no
        displacement.
        """
        tracks = track_detections(
            make_detections([stationary(0, 0, 40, jitter=1.0)], frames=40)
        )
        row = summarise(tracks).iloc[0]
        assert row["displacement_m"] < 5.0
        assert row["path_length_m"] > row["displacement_m"]

    def test_one_row_per_track(self):
        tracks = track_detections(
            make_detections(
                [straight(0, 0, 10, 0, 20), straight(0, 300, -10, 0, 20)], frames=20
            )
        )
        assert len(summarise(tracks)) == tracks["track_id"].nunique()
