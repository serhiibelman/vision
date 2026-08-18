
"""
Tests for the command line interface.

Stages that need a model or the real video are exercised through their own modules;
here the concern is argument wiring, calibration persistence and frame selection.
"""

from __future__ import annotations

import json

import pytest

from car_tracker.cli import build_parser, load_calibration, main, save_calibration
from car_tracker.geo import Calibration


class TestParser:
    @pytest.mark.parametrize(
        "command", ["telemetry", "calibrate", "detect", "track", "map", "all"]
    )
    def test_every_command_parses(self, command):
        args = build_parser().parse_args([command])
        assert args.command == command
        assert callable(args.func)

    def test_command_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_unknown_command_rejected(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["nope"])

    def test_frame_arguments(self):
        args = build_parser().parse_args(
            ["detect", "--stride", "3", "--start", "100", "--end", "200", "--limit", "10"]
        )
        assert (args.stride, args.start, args.end, args.limit) == (3, 100, 200, 10)

    def test_detector_defaults(self):
        args = build_parser().parse_args(["detect"])
        assert args.weights == "yolo11x-obb.pt"
        assert args.conf == 0.25

    def test_help_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["--help"])
        assert excinfo.value.code == 0
        assert "car-tracker" in capsys.readouterr().out


class TestFrameSelection:
    def _frames(self, argv, count):
        from car_tracker.cli import _frames

        return _frames(build_parser().parse_args(argv), count)

    def test_all_frames_by_default(self):
        assert self._frames(["detect"], 5) == [1, 2, 3, 4, 5]

    def test_stride(self):
        assert self._frames(["detect", "--stride", "2"], 6) == [1, 3, 5]

    def test_start_and_end_are_inclusive(self):
        assert self._frames(["detect", "--start", "2", "--end", "4"], 10) == [2, 3, 4]

    def test_limit_applies_before_range(self):
        """
        Limit caps how many frames are generated, then the range filters them.
        """
        assert self._frames(["detect", "--limit", "3"], 100) == [1, 2, 3]


class TestCalibrationPersistence:
    def test_round_trip(self, tmp_path):
        original = Calibration(
            gsd_scale=1.036, yaw_offset_deg=-0.39, yaw_sign=1, residual_m=0.29
        )
        path = save_calibration(original, tmp_path / "sub" / "calibration.json")
        assert load_calibration(path) == original

    def test_written_as_readable_json(self, tmp_path):
        path = save_calibration(Calibration(gsd_scale=1.5), tmp_path / "c.json")
        assert json.loads(path.read_text())["gsd_scale"] == 1.5

    def test_missing_file_gives_uncalibrated_defaults(self, tmp_path):
        assert load_calibration(tmp_path / "absent.json") == Calibration()

    def test_none_gives_uncalibrated_defaults(self):
        assert load_calibration(None) == Calibration()


class TestTelemetryCommand:
    def test_writes_csv_and_map(self, tmp_path, capsys):
        srt = tmp_path / "sample.SRT"
        srt.write_text(
            "\n\n".join(
                f"""{i}
00:00:0{i},000 --> 00:00:0{i},033
<font size="28">FrameCnt: {i}, DiffTime: 33ms
2024-11-24 17:38:0{i}.000
[iso: 100] [focal_len: 24.00] [dzoom_ratio: 1.00], [latitude: 48.26701{i}] \
[longitude: 25.91456{i}] [rel_alt: 102.2 abs_alt: 426.1] \
[gb_yaw: -65.8 gb_pitch: -90.0 gb_roll: 0.0] </font>"""
                for i in range(1, 4)
            ),
            encoding="utf-8",
        )
        out = tmp_path / "telemetry.csv"
        html = tmp_path / "flight.html"
        assert main(["telemetry", "--srt", str(srt), "--out", str(out), "--map", str(html)]) == 0
        assert out.exists()
        assert html.exists()
        assert "frames" in capsys.readouterr().out
