
# Car Path Tracking

Detect moving cars in nadir drone footage and plot their paths on a map.
Task 3 of the Python Engineer test assignment.

The drone's SRT sidecar carries per-frame GPS, altitude and gimbal angles, and the
gimbal points straight down for the entire flight. That makes pixel→ground a simple
scale-and-rotate, so vehicles detected in each frame can be placed on a real map.
Moving cars are then separated from parked ones **in geographic space**: a parked car
holds a fixed coordinate while the drone flies over it, and a moving car does not.

## Install

Requires Python ≥ 3.10 and an NVIDIA GPU for a full-length run (CPU works, but see
[runtime](#runtime)).

```bash
cd car_tracker
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip

# torch first — the CUDA vs CPU wheel cannot be declared in pyproject.toml
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CPU-only machine instead:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[detect,dev]"
```

Core install without the detector (permissive licences only): `pip install -e .`

Place `video2.MP4` and `video2.SRT` in `tech-assignment/`.

## Run

Everything at once:

```bash
cd car_tracker
car-tracker all
```

Or stage by stage — each reads files and writes files, so the slow detection pass runs
once while the map can be redrawn in seconds:

```bash
car-tracker telemetry --map outputs/flight_path.html   # SRT -> telemetry.csv
car-tracker calibrate                                  # -> calibration.json
car-tracker detect --stride 1                          # -> detections.csv   (slow)
car-tracker track                                      # -> tracks.csv
car-tracker map                                        # -> results/map.html
```

Useful flags: `--stride N` (every Nth frame), `--start` / `--end` (frame range),
`--limit N`, `--conf` (detector threshold), `--weights`.

A short segment on a CPU machine:

```bash
car-tracker detect --start 1000 --end 1300 --stride 3
car-tracker track && car-tracker map
```

## Output

| File | Contents |
|---|---|
| `results/map.html` | the deliverable — car paths on satellite imagery |
| `results/tracks.geojson` | moving-car paths as LineStrings with speed and distance |
| `outputs/telemetry.csv` | per-frame drone position, altitude, gimbal angles |
| `outputs/calibration.json` | measured pixel→ground transform |
| `outputs/detections.csv` | per-frame vehicle detections |
| `outputs/tracks.csv` | linked tracks with speed and heading |
| `outputs/track_features.csv` | per-track verdict, including why a track was rejected |

`outputs/` is regenerable and gitignored; `results/` is committed so the map can be
opened without a GPU or a copy of the 656 MB video.

The map has toggleable layers: moving cars, rejected (parked) tracks, the drone's
flight path, and satellite/street basemaps. Rejected tracks are kept visible on purpose
— if a genuine car is filtered out, that layer is where it shows.

## Results

Full video, every frame (GPU, ~12 min):

```
19,698 detections -> 313 tracks
moving                 76      <- manual count of the footage: 56
rejected: stationary  124
rejected: too_short    76
rejected: erratic      28
rejected: implausible_speed 5
rejected: too_slow      3
displacement       11 .. 190 m
average speed      46 km/h
```

The count both over- and under-reports, for two independent reasons: parked cars in dense
lots are joined into false paths, and cars in the blind altitude band are never detected.
Both are quantified below.

A 10-second window (frames 1000–1300, every 3rd frame), useful as a quick CPU check:

```
282 detections -> 15 tracks
moving             5
displacement      20 .. 32 m
median speed      34.6 km/h
```

Calibration accuracy: a static ground point projects to within **0.29 m** from two
different frames, measured over 35 frame pairs spread across the flight.

**Verified visually**: the plotted car paths lie on the actual roads when the map is viewed
over satellite imagery. That checks the whole chain externally — SRT parsing, calibration,
projection, tracking — rather than only self-consistently via the residual above.

The moving count over-reports — see [Known limitations](#known-limitations).

## Challenges encountered

**COCO-pretrained detectors do not work on nadir footage.** YOLO11x with COCO weights
reached ~29% recall and confidently boxed snowy building roofs as vehicles. The cause is
a domain gap — COCO is ground-level photography and has effectively never seen a car
from directly overhead. Switching to DOTA-trained weights (aerial imagery) took
precision to ~100% on the frames checked, with no threshold tuning.

**Library defaults silently destroy small objects.** At `imgsz=640` a 1920-wide frame is
downscaled 3×, shrinking a car from ~57 px to ~19 px. That alone dropped detections from
61 to 4 across a 20-frame sample. Tiling at native resolution was necessary before any
model could be judged fairly.

**The camera spec could not be trusted, and neither could the obvious workaround.**
`focal_len: 24.00` is ambiguous, so ground scale was measured from the footage instead
(GPS displacement versus pixel shift). An earlier attempt to check it by measuring car
lengths in detected boxes suggested the spec was 30% wrong; calibration showed it was
accurate to 3.6% and the *boxes* were the unreliable part. Assumption-free measurement
won over the assumption-laden one.

**Pixel-space tracking cannot work from a moving camera.** A parked car streaks across
the frame at the drone's speed while a car travelling alongside barely shifts, so pixel
distance is meaningless for association. Detections are geo-referenced *before* being
linked, and the gate is physical: a car cannot cover more than `max_speed × dt`.

**Most cars in this footage are parked.** Roughly twenty stationary vehicles per frame
against a couple driving, so a detector that finds everything has not answered the
question. The moving/parked split is geometric — straightness (displacement ÷ path
length) collapses for a car that jitters in place under GPS noise.

**Drone GPS drift is common-mode.** Drift in the drone's own fix displaces every
projected object together, making whole streets of parked cars appear to drift in
formation. Corrected by subtracting the median step across all tracks, which works
because most tracks are parked.

**Two bugs found by tests rather than by inspection.** `metres_to_latlon` used a
spherical Earth while distances used the WGS84 ellipsoid — a 0.1% bias that would have
put an unreachable floor under the calibration residual. And `max_coast_s` was enforced
after association instead of before, so a track could coast across an arbitrarily long
gap and then claim a distant detection.

## How it works

```
video2.SRT ──▶ telemetry ──▶ camera model ◀── calibration (GSD + yaw from GPS vs flow)
                                  │
video2.MP4 ──▶ detect ────────────┤  YOLO11 + DOTA weights, tiled 640px
              (per frame)         ▼
                            georeference ──▶ track (Hungarian + Kalman, in metres)
                                                  │
                                    moving/parked filter ──▶ map.html
```

| Stage | Module | Doc |
|---|---|---|
| Parse SRT | `telemetry.py` | [telemetry.md](docs/car_tracker/telemetry.md) |
| Frame access | `video.py` | [video.md](docs/car_tracker/video.md) |
| Pixel → coordinate | `geo.py` | [geo.md](docs/car_tracker/geo.md) |
| Measure the transform | `calibrate.py` | [calibrate.md](docs/car_tracker/calibrate.md) |
| Find vehicles | `detect.py` | [detect.md](docs/car_tracker/detect.md) |
| Link into paths | `track.py` | [track.md](docs/car_tracker/track.md) |
| Moving vs parked | `postprocess.py` | [postprocess.md](docs/car_tracker/postprocess.md) |
| Maps | `visualization.py` | [visualization.md](docs/car_tracker/visualization.md) |

Design rationale: [architecture.md](docs/car_tracker/architecture.md).
Why each choice was made, including the ones that turned out wrong:
[DECISIONS.md](DECISIONS.md). Progress: [PLAN.md](PLAN.md).

## Runtime

| Machine | Full 4979 frames |
|---|---|
| NVIDIA GPU | ~10–15 min |
| CPU (4 cores) | ~16 h — use `--stride` |

Detection dominates; every other stage takes seconds.

## Known limitations

**Detection has a blind altitude band.** Over roughly frames 3163–4142 (70–85 m altitude)
the detector averages **0.8 detections per frame**, against 4.4–5.8 elsewhere. Verified by
hand: frame 3620 contains at least nine clearly visible vehicles and one is detected.
Lowering the confidence threshold to 0.05 recovers none of them, so they are never
proposed rather than scored low. Cars in that stretch are largely absent from the map.
Details and the rejected two-model fix: [detect.md](docs/car_tracker/detect.md) and
DECISIONS.md D8.


**Moving cars are over-reported: 88 against a manual count of 56.** All the extra ones
trace to the same cause, and it is geometric rather than a tuning mistake.

| Quantity | Value |
|---|---|
| Car pitch in a parking lot, side by side | **~2.5 m** |
| Projection error | 0.3 m typical, **1.5 m during rapid yaw** |
| Detector centroid jitter | ~0.5–1 m |

The association gate cannot be tighter than our own measurement error, and that error is
comparable to the spacing between adjacent parked cars. So a track can hop to the
neighbouring vehicle, and because a parking row is collinear the resulting path is
perfectly straight with steadily growing displacement — the exact signature of a real car.

What the full-run data shows:

- 45 of 88 "moving" tracks travel within 30° of the drone's own heading, and 28 of those
  also match its speed to within ±40%. A track following the drone is a track walking
  along a row of parked cars as new ones enter the frame.
- They cluster in frames 4563–4979, the final descent over dense parking.
- Some contain single-frame steps of 25–35 m, i.e. ~1000 m/s.

Rejected tracks are kept in `outputs/track_features.csv` with the reason, and drawn as a
hidden map layer, so every classification can be audited rather than trusted.

Approaches tried and reverted, with measurements, are in
[DECISIONS.md](DECISIONS.md) D7 — worth reading before attempting a fix, since the two
obvious ones both cost more than they gained.

## Assumptions

- Flat ground at the projection plane; no terrain model, so relief adds error
- Camera is nadir — verified, `gb_pitch` is −90° in all 4979 frames
- No lens distortion correction; no calibration data exists for this camera
- GPS accurate to a few metres, so relative path shape is better than absolute position
- Detector box dimensions are **not** used as vehicle length (they are unreliable)

## Tests

```bash
cd car_tracker && pytest        # 287 tests
ruff check src tests
```

Detection tests stub the model, so no weights or GPU are needed.

## Licence

AGPL-3.0-or-later. The copyleft obligation comes from `ultralytics`, reachable only
through the optional `detect` extra; the core dependencies are all permissive
(BSD/MIT/Apache-2.0). `torchvision`'s Faster R-CNN (BSD) is a drop-in alternative if
that matters.
