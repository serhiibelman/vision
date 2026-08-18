# Architecture

Not yet implemented — this is the design.

## Input facts

Measured from `video2.MP4` / `video2.SRT`, not assumed.

| Fact | Consequence |
|---|---|
| `gb_pitch` = −90° for all 4979 frames | camera is pure nadir → pixel→ground is scale + rotation, not perspective |
| `focal_len` 24.00, `dzoom_ratio` 1.00, constant | intrinsics never change |
| `rel_alt` 102 m (frames 1–2883) → 61 m (end) | scale changes per frame; ~57 px cars for most of the video |
| `gb_yaw` −65.8° → 115.7° | camera rotates ~180° during flight |
| 1 SRT block per video frame | join on frame index, no interpolation |
| 1920×1080, 29.97 fps, 4979 frames, 166 s | GSD 8.0 → 4.8 cm/px |

Scene: snowy urban street. **~20 parked cars per frame, a couple moving.**

## The two real problems

**1. Moving vs parked.** The camera moves, so everything moves in image space.
Solved in geo space: a parked car holds a constant lat/lon while the drone flies
over it; a moving car translates. This is the core of the deliverable.

**2. Scale and yaw sign.** Every distance depends on them. Don't trust the spec —
`focal_len: 24.00` is ambiguous (35 mm-equivalent or actual). Self-calibrate:

```
GSD [m/px] = GPS displacement [m] / background pixel shift [px]
```

Validation gate: a static landmark must project to a constant lat/lon across
frames. Residual in metres = the accuracy figure. **Nothing downstream is built
until this passes** — every later stage inherits this error.

## Pipeline

```
video2.SRT ──▶ telemetry ──▶ camera model ◀── self-calibration
                                  │
video2.MP4 ──▶ detect ────────────┤
              (YOLO11 + DOTA,     │
               tiled 640)         ▼
                            georeference ──▶ track (metres)
                                                  │
                                                  ▼
                                    static/moving filter ──▶ map.html
```

Stage boundaries are files: detection over 4979 frames is slow and must run once
while the map is re-iterated in seconds.

## Key choices

| Choice | Reason |
|---|---|
| DOTA-trained weights, tiled at native res | COCO detects buildings as cars; downscaling shrinks cars to 19 px. See D1. |
| Georeference *before* tracking | pixel distance is not a valid association gate — a parked car streaks across frame at drone speed |
| Hungarian + constant-velocity Kalman in ENU metres | physical gate (~3 m per 0.1 s); velocity as direct output; no AGPL |
| Accept ~80% per-frame recall | a car spans ~300 frames; tracking closes gaps |

## Assumptions

- Flat ground at the projection plane (no DEM) → terrain relief adds error
- Negligible lens distortion near image centre (no calibration data available)
- GPS accurate to a few metres → relative path shape better than absolute position

## Known failure modes

**Dense parking lots.** The largest known inaccuracy: 88 moving cars reported against a
manual count of 56. Cars sit at ~2.5 m pitch while projection error reaches 1.5 m during
rapid yaw, so the association gate cannot exclude the neighbouring vehicle. A parking row is
collinear, so a hop chain looks perfectly straight with growing displacement. Root cause,
measurements and rejected fixes: DECISIONS.md D7.

**Parallax.** At nadir, an object at height *h* displaces an extra ≈ `h/alt` when
the drone translates — a 15 m building at 100 m alt shifts ~15% more than the
ground. Rooftops therefore look like moving objects. Rejected by the geo-space
filter (a building oscillates around a point, a car travels a monotonic path), not
by image-space thresholds.
