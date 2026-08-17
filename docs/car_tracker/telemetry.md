
# telemetry

`car_tracker/src/car_tracker/telemetry.py` — parses the DJI SRT sidecar into a
per-frame table. Implemented.

## Purpose

Converts the SRT into the data needed to map pixels to real-world coordinates.
Without it, car paths exist only in pixels inside a moving frame.

## API

```python
from car_tracker.telemetry import load, summarise

df = load("video2.SRT", expected_frames=4979)   # parse + validate
print(summarise(df))
```

| Function | Does |
|---|---|
| `load(path, expected_frames=None)` | parse + validate. Normal entry point. |
| `parse_srt(path)` | parse only → DataFrame |
| `parse_block(block)` | one SRT block → dict |
| `validate(df, expected_frames=None)` | raises `TelemetryError` on bad data |
| `ground_sample_distance(rel_alt, width=1920)` | metres per pixel |
| `summarise(df)` | text summary for CLI |

## Output columns

| Column | Unit | Source |
|---|---|---|
| `frame` | 1-based int | `FrameCnt` |
| `timestamp` | datetime | block line 2 |
| `t_sec` | seconds from start | derived |
| `lat`, `lon` | degrees | `latitude`, `longitude` |
| `rel_alt`, `abs_alt` | metres | `rel_alt`, `abs_alt` |
| `yaw`, `pitch`, `roll` | degrees | `gb_yaw`, `gb_pitch`, `gb_roll` |
| `focal_len` | mm | `focal_len` |
| `dzoom` | ratio | `dzoom_ratio` |
| `gsd` | metres/pixel | derived from `rel_alt` |

## How it works

1. Split file on blank lines, keep blocks containing `FrameCnt`.
2. Match each field with its own regex — field order and the `<font>` wrapper vary
   between firmware versions, so position-based parsing is unsafe.
3. Sort by `frame`; derive `t_sec` and `gsd`.
4. Validate.

## Validation

Fails loudly — every later stage trusts this table.

| Check | Rejects |
|---|---|
| frame numbers contiguous | dropped blocks that would misalign the frame join |
| `len(df) == video frames` | telemetry/video mismatch |
| timestamps monotonic | corrupt ordering |
| `pitch` within 2° of −90° | off-nadir camera → the scale+rotation model breaks |
| `lat`/`lon` non-zero, in range | lost GPS fix |
| `rel_alt > 0` | invalid scale |

## Measured on video2.SRT

```
frames     4979  (1..4979)
duration   166.06 s
latitude   48.263933 .. 48.268341
longitude  25.900229 .. 25.919477
rel_alt    60.6 .. 102.2 m
yaw        -172.7 .. 173.1 deg
pitch      -90.0 .. -89.9 deg     <- nadir confirmed, all frames
gsd        4.7 .. 8.0 cm/px
```

## Caveat

`gsd` is provisional. It assumes `focal_len: 24.00` is a 35 mm equivalent on a
36 mm-wide sensor, which is unverified. Step 3 replaces it with calibration against
observed ground motion.

## Design note

Functions returning a DataFrame, not a class — pure data transform with no state.
Downstream joins on `frame` in one vectorized pass rather than per-frame lookups.

## Tests

`car_tracker/tests/test_telemetry.py` — 28 tests, fixture of 3 real SRT blocks.
Covers field extraction, negative values, missing fields, derived columns, GSD
scaling, and every validation rule.
