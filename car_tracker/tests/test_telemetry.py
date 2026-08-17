"""
Tests for SRT telemetry parsing and validation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from car_tracker.telemetry import (
    COLUMNS,
    TelemetryError,
    ground_sample_distance,
    load,
    parse_block,
    parse_srt,
    validate,
)

# Two real blocks copied verbatim from video2.SRT, plus a third with the frame
# number advanced, so ordering and derived columns can be checked.
FIXTURE = """1
00:00:00,000 --> 00:00:00,033
<font size="28">FrameCnt: 1, DiffTime: 33ms
2024-11-24 17:38:03.576
[iso: 100] [shutter: 1/798.21] [fnum: 2.8] [ev: 0] [color_md : default] \
[ae_meter_md: 1] [focal_len: 24.00] [dzoom_ratio: 1.00], [latitude: 48.267013] \
[longitude: 25.914562] [rel_alt: 102.229 abs_alt: 426.185] \
[gb_yaw: -65.8 gb_pitch: -89.9 gb_roll: 0.0] </font>

2
00:00:00,033 --> 00:00:00,066
<font size="28">FrameCnt: 2, DiffTime: 33ms
2024-11-24 17:38:03.612
[iso: 100] [shutter: 1/798.21] [fnum: 2.8] [ev: 0] [color_md : default] \
[ae_meter_md: 1] [focal_len: 24.00] [dzoom_ratio: 1.00], [latitude: 48.267014] \
[longitude: 25.914563] [rel_alt: 102.231 abs_alt: 426.188] \
[gb_yaw: -65.7 gb_pitch: -90.0 gb_roll: 0.0] </font>

3
00:00:00,066 --> 00:00:00,099
<font size="28">FrameCnt: 3, DiffTime: 33ms
2024-11-24 17:38:03.676
[iso: 100] [shutter: 1/798.21] [fnum: 2.8] [ev: 0] [color_md : default] \
[ae_meter_md: 1] [focal_len: 24.00] [dzoom_ratio: 1.00], [latitude: 48.267015] \
[longitude: 25.914564] [rel_alt: 102.500 abs_alt: 426.456] \
[gb_yaw: -65.6 gb_pitch: -90.0 gb_roll: 0.0] </font>
"""


@pytest.fixture
def srt_file(tmp_path):
    path = tmp_path / "sample.SRT"
    path.write_text(FIXTURE, encoding="utf-8")
    return path


@pytest.fixture
def telemetry(srt_file):
    return parse_srt(srt_file)


class TestParseBlock:
    def test_extracts_every_field(self):
        block = FIXTURE.split("\n\n")[0]
        row = parse_block(block)
        assert row["frame"] == 1
        assert row["lat"] == pytest.approx(48.267013)
        assert row["lon"] == pytest.approx(25.914562)
        assert row["rel_alt"] == pytest.approx(102.229)
        assert row["abs_alt"] == pytest.approx(426.185)
        assert row["yaw"] == pytest.approx(-65.8)
        assert row["pitch"] == pytest.approx(-89.9)
        assert row["roll"] == pytest.approx(0.0)
        assert row["focal_len"] == pytest.approx(24.0)
        assert row["dzoom"] == pytest.approx(1.0)
        assert row["timestamp"] == "2024-11-24 17:38:03.576"

    def test_negative_values_keep_their_sign(self):
        """
        Yaw and pitch are routinely negative; a greedy digit match would drop it.
        """
        row = parse_block(FIXTURE.split("\n\n")[0])
        assert row["yaw"] < 0
        assert row["pitch"] < 0

    @pytest.mark.parametrize("field", ["latitude", "gb_yaw", "rel_alt", "FrameCnt"])
    def test_missing_field_raises(self, field):
        block = FIXTURE.split("\n\n")[0].replace(field, "removed")
        with pytest.raises(TelemetryError, match="missing"):
            parse_block(block)

    def test_missing_timestamp_raises(self):
        block = FIXTURE.split("\n\n")[0].replace("2024-11-24 17:38:03.576", "")
        with pytest.raises(TelemetryError, match="timestamp"):
            parse_block(block)


class TestParseSrt:
    def test_row_per_block_with_expected_columns(self, telemetry):
        assert len(telemetry) == 3
        assert list(telemetry.columns) == COLUMNS

    def test_sorted_by_frame(self, telemetry):
        assert telemetry["frame"].tolist() == [1, 2, 3]

    def test_timestamp_parsed_to_datetime(self, telemetry):
        assert pd.api.types.is_datetime64_any_dtype(telemetry["timestamp"])

    def test_t_sec_is_relative_to_first_frame(self, telemetry):
        assert telemetry["t_sec"].iloc[0] == 0.0
        assert telemetry["t_sec"].iloc[1] == pytest.approx(0.036, abs=1e-6)

    def test_gsd_grows_with_altitude(self, telemetry):
        """
        Higher flight means each pixel covers more ground.
        """
        assert telemetry["gsd"].iloc[2] > telemetry["gsd"].iloc[0]

    def test_no_blocks_raises(self, tmp_path):
        empty = tmp_path / "empty.SRT"
        empty.write_text("1\n00:00:00,000 --> 00:00:00,033\nno telemetry here\n")
        with pytest.raises(TelemetryError, match="no telemetry blocks"):
            parse_srt(empty)


class TestGroundSampleDistance:
    def test_known_altitude(self):
        """
        ~8 cm/px at 102 m for a 24 mm-equivalent lens on a 1920 px frame.
        """
        assert ground_sample_distance(102.229) == pytest.approx(0.0799, abs=5e-4)

    def test_scales_linearly(self):
        assert ground_sample_distance(120.0) == pytest.approx(
            2 * ground_sample_distance(60.0)
        )

    def test_accepts_arrays(self):
        result = ground_sample_distance(pd.Series([61.0, 102.0]).to_numpy())
        assert result[1] > result[0]


class TestValidate:
    def test_accepts_good_data(self, telemetry):
        validate(telemetry)

    def test_frame_count_mismatch_raises(self, telemetry):
        with pytest.raises(TelemetryError, match="misaligned"):
            validate(telemetry, expected_frames=4979)

    def test_matching_frame_count_passes(self, telemetry):
        validate(telemetry, expected_frames=3)

    def test_empty_raises(self):
        with pytest.raises(TelemetryError, match="empty"):
            validate(pd.DataFrame(columns=COLUMNS))

    def test_frame_gap_raises(self, telemetry):
        gapped = telemetry.drop(index=1).reset_index(drop=True)
        with pytest.raises(TelemetryError, match="not contiguous"):
            validate(gapped)

    def test_non_monotonic_timestamps_raise(self, telemetry):
        telemetry.loc[2, "timestamp"] = telemetry.loc[0, "timestamp"]
        with pytest.raises(TelemetryError, match="monotonic"):
            validate(telemetry)

    def test_off_nadir_pitch_raises(self, telemetry):
        """
        A tilted camera invalidates the scale+rotation ground model.
        """
        telemetry.loc[1, "pitch"] = -45.0
        with pytest.raises(TelemetryError, match="off-nadir"):
            validate(telemetry)

    def test_slight_pitch_wobble_is_tolerated(self, telemetry):
        telemetry.loc[1, "pitch"] = -88.5
        validate(telemetry)

    def test_zeroed_gps_raises(self, telemetry):
        telemetry.loc[1, "lat"] = 0.0
        with pytest.raises(TelemetryError, match="lat"):
            validate(telemetry)

    def test_non_positive_altitude_raises(self, telemetry):
        telemetry.loc[1, "rel_alt"] = 0.0
        with pytest.raises(TelemetryError, match="altitude"):
            validate(telemetry)


class TestLoad:
    def test_parses_and_validates(self, srt_file):
        frame = load(srt_file, expected_frames=3)
        assert len(frame) == 3

    def test_propagates_validation_error(self, srt_file):
        with pytest.raises(TelemetryError):
            load(srt_file, expected_frames=999)
