# Plan — Task 3: Car Path Tracking

Track moving cars in drone footage and plot their paths on a map.
Inputs: `tech-assignment/video2.MP4` + `video2.SRT`.

| # | Step | Output | Status |
|---|---|---|---|
| 1 | Parse SRT telemetry | `telemetry.csv` | ✅ `telemetry.py` |
| 2 | Plot drone flight path on map | `flight_path.html` | ✅ `visualization.py` |
| 3 | Calibrate pixel→lat/lon (GSD + yaw), verify a static point stays put | residual in metres — **GATE** | ✅ **passed, 0.29 m** |
| 4 | Detect vehicles per frame | `detections.csv` | next |
| 5 | Project to lat/lon, track in metres | `tracks.csv` | |
| 6 | Filter moving vs parked | `tracks.geojson` | |
| 7 | Final map + README | deliverables | |

**Step 3 was the gate** — every later stage inherits its error. Passed: a static
ground point projects to within 0.29 m from two different frames, and the yaw
convention is resolved (`rotation = +1 * yaw − 0.39°`).

Design rationale: [`docs/car_tracker/architecture.md`](docs/car_tracker/architecture.md)
Decision log: [`DECISIONS.md`](DECISIONS.md)

## Status

Steps 1–3 implemented. 131 tests passing, ruff clean.

| Module | Doc |
|---|---|
| `telemetry.py` | [telemetry.md](docs/car_tracker/telemetry.md) |
| `visualization.py` | [visualization.md](docs/car_tracker/visualization.md) |
| `geo.py` | [geo.md](docs/car_tracker/geo.md) |
| `calibrate.py` | [calibrate.md](docs/car_tracker/calibrate.md) |
| `video.py` | [video.md](docs/car_tracker/video.md) |

Detector decided: YOLO11 + DOTA weights (see D1). Spike in `car_tracker/spikes/`.
