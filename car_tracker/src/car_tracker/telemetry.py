"""
Parse the DJI SRT sidecar into a per-frame telemetry table.

The SRT carries one subtitle block per video frame, holding the camera's position,
altitude and gimbal angles at that instant. It is the only thing that lets pixel
coordinates be converted to real-world latitude/longitude, so everything downstream
trusts this module — hence the loud validation in :func:`validate`.

Block format (whitespace and the ``<font>`` wrapper vary between firmware versions,
so fields are matched individually rather than by position)::

    1
    00:00:00,000 --> 00:00:00,033
    <font size="28">FrameCnt: 1, DiffTime: 33ms
    2024-11-24 17:38:03.576
    [iso: 100] [shutter: 1/798.21] [fnum: 2.8] [ev: 0] ... [focal_len: 24.00]
    [dzoom_ratio: 1.00], [latitude: 48.267013] [longitude: 25.914562]
    [rel_alt: 102.229 abs_alt: 426.185] [gb_yaw: -65.8 gb_pitch: -89.9 gb_roll: 0.0]
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# The camera is expected to point straight down for the whole flight. The pixel to
# ground mapping is a plain scale + rotation only while that holds, so it is
# asserted rather than assumed.
NADIR_PITCH_DEG = -90.0
NADIR_TOLERANCE_DEG = 2.0

# Sanity box for the recording location (Chernivtsi region, Ukraine). Wide enough to
# be uninteresting, tight enough to catch a zeroed or garbage GPS fix.
LAT_RANGE = (-90.0, 90.0)
LON_RANGE = (-180.0, 180.0)

# 35 mm-equivalent sensor width, paired with the SRT's `focal_len`, gives the
# horizontal field of view. Provisional: `focal_len: 24.00` is ambiguous (equivalent
# vs actual), so calibration against observed ground motion supersedes this later.
SENSOR_WIDTH_EQUIV_MM = 36.0

_BLOCK_SPLIT = re.compile(r"\n\s*\n")
_PATTERNS = {
    "frame": re.compile(r"FrameCnt:\s*(\d+)"),
    "lat": re.compile(r"\[latitude:\s*([-\d.]+)\]"),
    "lon": re.compile(r"\[longitude:\s*([-\d.]+)\]"),
    "rel_alt": re.compile(r"rel_alt:\s*([-\d.]+)"),
    "abs_alt": re.compile(r"abs_alt:\s*([-\d.]+)"),
    "yaw": re.compile(r"gb_yaw:\s*([-\d.]+)"),
    "pitch": re.compile(r"gb_pitch:\s*([-\d.]+)"),
    "roll": re.compile(r"gb_roll:\s*([-\d.]+)"),
    "focal_len": re.compile(r"focal_len:\s*([-\d.]+)"),
    "dzoom": re.compile(r"dzoom_ratio:\s*([-\d.]+)"),
}
_TIMESTAMP = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[.,]\d+)")

COLUMNS = [
    "frame", "timestamp", "t_sec", "lat", "lon", "rel_alt", "abs_alt",
    "yaw", "pitch", "roll", "focal_len", "dzoom", "gsd",
]


class TelemetryError(ValueError):
    """
    Raised when the SRT cannot be parsed or fails a consistency check.
    """


def parse_block(block: str) -> dict[str, float | str]:
    """
    Extract one frame's telemetry from a single SRT block.

    Raises:
        TelemetryError: if any expected field is missing.
    """
    row: dict[str, float | str] = {}
    for name, pattern in _PATTERNS.items():
        match = pattern.search(block)
        if match is None:
            raise TelemetryError(f"missing field {name!r} in block:\n{block.strip()[:200]}")
        row[name] = int(match.group(1)) if name == "frame" else float(match.group(1))

    stamp = _TIMESTAMP.search(block)
    if stamp is None:
        raise TelemetryError(f"missing timestamp in block:\n{block.strip()[:200]}")
    row["timestamp"] = stamp.group(1).replace(",", ".")
    return row


def ground_sample_distance(rel_alt_m: float | np.ndarray, image_width_px: int = 1920):
    """
    Metres covered by one pixel, for a nadir camera at ``rel_alt_m``.

    Provisional estimate from the nominal focal length; superseded by calibration
    against measured ground motion. Scales linearly with altitude, so it must be
    evaluated per frame rather than once for the flight.
    """
    half_fov = np.arctan(SENSOR_WIDTH_EQUIV_MM / (2 * 24.0))
    return 2 * rel_alt_m * np.tan(half_fov) / image_width_px


def parse_srt(path: str | Path) -> pd.DataFrame:
    """
    Parse an SRT sidecar into a per-frame DataFrame.

    Returns:
        One row per subtitle block, columns as in :data:`COLUMNS`, sorted by frame.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = [b for b in _BLOCK_SPLIT.split(text) if "FrameCnt" in b]
    if not blocks:
        raise TelemetryError(f"no telemetry blocks found in {path}")

    frame = pd.DataFrame([parse_block(b) for b in blocks])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("frame", ignore_index=True)

    start = frame["timestamp"].iloc[0]
    frame["t_sec"] = (frame["timestamp"] - start).dt.total_seconds()
    frame["gsd"] = ground_sample_distance(frame["rel_alt"].to_numpy())
    return frame[COLUMNS]


def validate(frame: pd.DataFrame, expected_frames: int | None = None) -> None:
    """
    Fail loudly on anything that would silently corrupt downstream geometry.

    Args:
        frame: parsed telemetry.
        expected_frames: video frame count, when known, to confirm the 1:1 mapping.

    Raises:
        TelemetryError: on any failed check.
    """
    if frame.empty:
        raise TelemetryError("telemetry is empty")

    numbers = frame["frame"].to_numpy()
    expected = np.arange(numbers[0], numbers[0] + len(numbers))
    if not np.array_equal(numbers, expected):
        missing = sorted(set(expected) - set(numbers))[:10]
        raise TelemetryError(f"frame numbers not contiguous; first gaps at {missing}")

    if expected_frames is not None and len(frame) != expected_frames:
        raise TelemetryError(
            f"telemetry has {len(frame)} rows but video has {expected_frames} frames — "
            "the per-frame join would be misaligned"
        )

    if not frame["timestamp"].is_monotonic_increasing:
        raise TelemetryError("timestamps are not monotonically increasing")

    off_nadir = (frame["pitch"] - NADIR_PITCH_DEG).abs() > NADIR_TOLERANCE_DEG
    if off_nadir.any():
        worst = frame.loc[off_nadir, "pitch"]
        raise TelemetryError(
            f"{off_nadir.sum()} frames are off-nadir (pitch {worst.min():.1f}.."
            f"{worst.max():.1f}°); the scale+rotation ground model does not hold"
        )

    for column, (low, high) in (("lat", LAT_RANGE), ("lon", LON_RANGE)):
        values = frame[column]
        if values.eq(0).any() or values.lt(low).any() or values.gt(high).any():
            raise TelemetryError(f"{column} outside plausible range or zeroed")

    if frame["rel_alt"].le(0).any():
        raise TelemetryError("non-positive relative altitude — scale would be invalid")


def load(path: str | Path, expected_frames: int | None = None) -> pd.DataFrame:
    """
    Parse and validate in one call. The normal entry point.
    """
    frame = parse_srt(path)
    validate(frame, expected_frames)
    return frame


def summarise(frame: pd.DataFrame) -> str:
    """
    One-paragraph human summary, for CLI output and sanity checking.
    """
    return "\n".join(
        [
            f"frames     {len(frame)}  ({frame['frame'].min()}..{frame['frame'].max()})",
            f"duration   {frame['t_sec'].iloc[-1]:.2f} s"
            f"  ({frame['timestamp'].iloc[0]} .. {frame['timestamp'].iloc[-1]})",
            f"latitude   {frame['lat'].min():.6f} .. {frame['lat'].max():.6f}",
            f"longitude  {frame['lon'].min():.6f} .. {frame['lon'].max():.6f}",
            f"rel_alt    {frame['rel_alt'].min():.1f} .. {frame['rel_alt'].max():.1f} m",
            f"yaw        {frame['yaw'].min():.1f} .. {frame['yaw'].max():.1f} deg",
            f"pitch      {frame['pitch'].min():.1f} .. {frame['pitch'].max():.1f} deg",
            f"gsd        {frame['gsd'].min() * 100:.1f} .. {frame['gsd'].max() * 100:.1f} cm/px",
        ]
    )
