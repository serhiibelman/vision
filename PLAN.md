# Plan — Task 3: Car Path Tracking

Track moving cars in drone footage and plot their paths on a map.
Inputs: `tech-assignment/video2.MP4` + `video2.SRT`.

| # | Step | Output |
|---|---|---|
| 1 | Parse SRT telemetry | `telemetry.csv` |
| 2 | Plot drone flight path on map | `map.html` — first working artifact |
| 3 | Calibrate pixel→lat/lon (GSD + yaw), verify a static landmark stays put | residual in metres — **GATE** |
| 4 | Detect vehicles per frame | `detections.csv` |
| 5 | Project to lat/lon, track in metres | `tracks.csv` |
| 6 | Filter moving vs parked | `tracks.geojson` |
| 7 | Final map + README | deliverables |

**Step 3 is the gate.** Nothing after it is worth building until a static landmark
holds a constant position across frames — every later stage inherits that error.

Design rationale: [`docs/car_tracker/architecture.md`](docs/car_tracker/architecture.md)
Decision log: [`DECISIONS.md`](DECISIONS.md)

## Status

Nothing implemented. Scaffolding + detector spike only (`car_tracker/spikes/`).

Detector decided: YOLO11 + DOTA weights (see D1).
