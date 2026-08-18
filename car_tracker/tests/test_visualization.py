"""
Tests for map rendering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from car_tracker.visualization import add_flight_path, base_map, path_length_m, render_flight_path


@pytest.fixture
def telemetry():
    """
    Five frames moving north-east while descending.
    """
    n = 5
    return pd.DataFrame(
        {
            "frame": np.arange(1, n + 1),
            "timestamp": pd.date_range("2024-11-24 17:38:03", periods=n, freq="33ms"),
            "t_sec": np.arange(n) * 0.033,
            "lat": 48.2670 + np.arange(n) * 1e-4,
            "lon": 25.9145 + np.arange(n) * 1e-4,
            "rel_alt": np.linspace(102.0, 61.0, n),
            "abs_alt": np.linspace(426.0, 385.0, n),
            "yaw": np.linspace(-65.8, 115.7, n),
            "pitch": np.full(n, -90.0),
            "roll": np.zeros(n),
            "focal_len": np.full(n, 24.0),
            "dzoom": np.ones(n),
            "gsd": np.linspace(0.08, 0.048, n),
        }
    )


class TestPathLength:
    def test_known_distance(self):
        """
        One degree of latitude is ~111 km.
        """
        lat = np.array([48.0, 49.0])
        lon = np.array([25.0, 25.0])
        assert path_length_m(lat, lon) == pytest.approx(111_200, rel=0.01)

    def test_sums_segments(self):
        straight = path_length_m(np.array([48.0, 48.02]), np.array([25.0, 25.0]))
        split = path_length_m(np.array([48.0, 48.01, 48.02]), np.array([25.0, 25.0, 25.0]))
        assert split == pytest.approx(straight, rel=1e-6)

    @pytest.mark.parametrize("n", [0, 1])
    def test_degenerate_input_is_zero(self, n):
        assert path_length_m(np.zeros(n), np.zeros(n)) == 0.0


class TestBaseMap:
    def test_centres_on_mean(self, telemetry):
        fmap = base_map(telemetry["lat"].to_numpy(), telemetry["lon"].to_numpy())
        assert fmap.location == pytest.approx(
            [telemetry["lat"].mean(), telemetry["lon"].mean()]
        )

    def test_has_both_layers(self, telemetry):
        fmap = base_map(telemetry["lat"].to_numpy(), telemetry["lon"].to_numpy())
        names = [c.tile_name for c in fmap._children.values() if hasattr(c, "tile_name")]
        assert "Satellite" in names
        assert "Street" in names


class TestRender:
    def test_writes_standalone_html(self, telemetry, tmp_path):
        out = render_flight_path(telemetry, tmp_path / "sub" / "flight.html")
        assert out.exists()
        html = out.read_text()
        assert "leaflet" in html.lower()
        assert "48.267" in html

    def test_creates_missing_parent_dirs(self, telemetry, tmp_path):
        out = render_flight_path(telemetry, tmp_path / "a" / "b" / "flight.html")
        assert out.parent.is_dir()

    def test_start_and_end_markers_present(self, telemetry, tmp_path):
        html = render_flight_path(telemetry, tmp_path / "f.html").read_text()
        assert "Start" in html
        assert "End" in html

    def test_step_decimates_polyline(self, telemetry, tmp_path):
        """
        A larger step must emit fewer vertices, not fail.
        """
        dense = render_flight_path(telemetry, tmp_path / "d.html", step=1).read_text()
        sparse = render_flight_path(telemetry, tmp_path / "s.html", step=4).read_text()
        assert len(sparse) < len(dense)

    def test_single_row_does_not_crash(self, telemetry, tmp_path):
        out = render_flight_path(telemetry.head(1), tmp_path / "one.html")
        assert out.exists()


class TestAddFlightPath:
    def test_adds_a_feature_group(self, telemetry):
        fmap = base_map(telemetry["lat"].to_numpy(), telemetry["lon"].to_numpy())
        before = len(fmap._children)
        add_flight_path(fmap, telemetry)
        assert len(fmap._children) > before


class TestCarTracks:
    @pytest.fixture
    def tracks(self):
        """
        Two cars driving east, one fast one slow.
        """
        rows = []
        for track_id, speed in ((1, 1e-4), (2, 4e-5)):
            for i in range(6):
                rows.append(
                    {
                        "track_id": track_id,
                        "frame": i + 1,
                        "t_sec": i / 30.0,
                        "lat": 48.2670 + track_id * 1e-4,
                        "lon": 25.9145 + i * speed,
                        "speed_mps": 12.0 if track_id == 1 else 5.0,
                        "conf": 0.8,
                    }
                )
        return pd.DataFrame(rows)

    @pytest.fixture
    def features(self):
        return pd.DataFrame(
            {
                "track_id": [1, 2, 3],
                "displacement_m": [40.0, 16.0, 1.0],
                "duration_s": [0.17, 0.17, 0.5],
                "median_speed_mps": [12.0, 5.0, 0.1],
                "is_moving": [True, True, False],
                "verdict": ["moving", "moving", "stationary"],
            }
        )

    def test_draws_a_polyline_per_track(self, tracks, features):
        from car_tracker.visualization import add_car_tracks, base_map

        fmap = base_map(tracks["lat"].to_numpy(), tracks["lon"].to_numpy())
        before = len(fmap._children)
        add_car_tracks(fmap, tracks, features)
        assert len(fmap._children) > before

    def test_renders_html_with_speed_labels(self, tracks, features, tmp_path):
        from car_tracker.visualization import render_tracks

        html = render_tracks(tracks, features, tmp_path / "tracks.html").read_text()
        assert "km/h" in html
        assert "car 1" in html

    def test_single_point_track_is_skipped(self, features, tmp_path):
        from car_tracker.visualization import render_tracks

        one = pd.DataFrame(
            {"track_id": [1], "t_sec": [0.0], "lat": [48.267], "lon": [25.9145],
             "speed_mps": [0.0]}
        )
        assert render_tracks(one, features, tmp_path / "one.html").exists()

    def test_rejected_layer_is_added_when_requested(self, tracks, features, tmp_path):
        from car_tracker.visualization import render_tracks

        rejected = tracks.assign(track_id=3)
        html = render_tracks(
            tracks, features, tmp_path / "r.html", all_tracks=pd.concat([tracks, rejected])
        ).read_text()
        assert "rejected" in html.lower()

    def test_flight_path_included_when_telemetry_given(self, tracks, features, telemetry, tmp_path):
        from car_tracker.visualization import render_tracks

        html = render_tracks(
            tracks, features, tmp_path / "f.html", telemetry=telemetry
        ).read_text()
        assert "Drone flight path" in html

    def test_empty_tracks_with_telemetry_still_renders(self, features, telemetry, tmp_path):
        from car_tracker.visualization import render_tracks

        empty = pd.DataFrame(columns=["track_id", "t_sec", "lat", "lon", "speed_mps"])
        assert render_tracks(empty, features, tmp_path / "e.html", telemetry=telemetry).exists()

    def test_nothing_to_draw_raises(self, features, tmp_path):
        from car_tracker.visualization import render_tracks

        empty = pd.DataFrame(columns=["track_id", "t_sec", "lat", "lon", "speed_mps"])
        with pytest.raises(ValueError, match="nothing to draw"):
            render_tracks(empty, features, tmp_path / "x.html")
