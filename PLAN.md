# Plan — Task 3: Car Path Tracking

Track moving cars in drone footage and plot their paths on a map.
Inputs: `tech-assignment/video2.MP4` + `video2.SRT`.

| # | Step | Output | Status |
|---|---|---|---|
| 1 | Parse SRT telemetry | `telemetry.csv` | ✅ `telemetry.py` |
| 2 | Plot drone flight path on map | `flight_path.html` | ✅ `visualization.py` |
| 3 | Calibrate pixel→lat/lon (GSD + yaw), verify a static point stays put | residual in metres — **GATE** | ✅ **passed, 0.29 m** |
| 4 | Detect vehicles per frame | `detections.csv` | ✅ `detect.py` |
| 5 | Project to lat/lon, track in metres | `tracks.csv` | ✅ `track.py` |
| 6 | Filter moving vs parked | `tracks.geojson` | ✅ `postprocess.py` |
| 7 | Final map + README | deliverables | ✅ `cli.py`, `README.md` |

**Step 3 was the gate** — every later stage inherits its error. Passed: a static
ground point projects to within 0.29 m from two different frames, and the yaw
convention is resolved (`rotation = +1 * yaw − 0.39°`).

Design rationale: [`docs/car_tracker/architecture.md`](docs/car_tracker/architecture.md)
Decision log: [`DECISIONS.md`](DECISIONS.md)

## Status

All seven steps implemented. **264 tests passing, ruff clean.** Pipeline runs end to
end via `car-tracker all`.

| Module | Doc |
|---|---|
| `telemetry.py` | [telemetry.md](docs/car_tracker/telemetry.md) |
| `video.py` | [video.md](docs/car_tracker/video.md) |
| `geo.py` | [geo.md](docs/car_tracker/geo.md) |
| `calibrate.py` | [calibrate.md](docs/car_tracker/calibrate.md) |
| `detect.py` | [detect.md](docs/car_tracker/detect.md) |
| `track.py` | [track.md](docs/car_tracker/track.md) |
| `postprocess.py` | [postprocess.md](docs/car_tracker/postprocess.md) |
| `visualization.py` | [visualization.md](docs/car_tracker/visualization.md) |
| `cli.py` | [cli.md](docs/car_tracker/cli.md) |

Detector decided: YOLO11 + DOTA weights (see D1). Spike in `car_tracker/spikes/`.

## Remaining before submission

| # | Item | Why it matters | Blocker? |
|---|---|---|---|
| 1 | **Full-video run on the GPU box** | current results cover one 10 s window (frames 1000–1300). The deliverable should cover all 166 s. `car-tracker all` — ~10–15 min on GPU, ~16 h on CPU | **yes** |
| 2 | **Visual verification of `results/map.html`** | confirms paths lie on roads. Nobody has looked yet | **yes** |
| 3 | Commit `results/` artifacts | so a reviewer without a GPU or the 656 MB video can open the map | yes |
| 4 | "Challenges encountered" section in README | the assignment asks for it explicitly | yes |
| 5 | Verify track 9 | 25.9 m in 1.2 s implies 78 km/h but filtered speed says 35 km/h, on 6 observations — possible association error | no |
| 6 | Annotated demo clip | not required by the assignment; most persuasive evidence detection works | no |
| 7 | `uv.lock` / `requirements-lock.txt` | exact reproducibility | no |

## Assignment deliverables checklist

From the PDF, Task 3:

| Required | Status |
|---|---|
| Code to detect cars, track their paths, plot them on a map | ✅ |
| A map visualization showing the paths of all detected cars | ⚠️ done for a 10 s window, not the full video |
| Documentation of approach, methods and assumptions | ✅ README + 9 module docs + architecture |
| Clean, modular code with comments and docstrings | ✅ |
| Instructions on how to run | ✅ README |
| Challenges encountered | ⚠️ in DECISIONS.md, not summarised in README |
| GitHub repo or zip with clear directory structure | ⚠️ commits exist; `results/` not yet committed |

## Known caveats

- Detection recall is ~80% per frame, which is fine — a car spans hundreds of frames
  and tracking closes the gaps. Never measured rigorously (2 frames hand-labelled).
- Detector box dimensions are unreliable as vehicle length (D5). Not used anywhere.
- Cross-pass re-identification is out of scope: the flight is out-and-back, so a car
  seen on both passes gets two track IDs.
- Flat-ground assumption, no lens distortion correction, no terrain model.
