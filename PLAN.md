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

Design rationale: [`car_tracker/docs/architecture.md`](car_tracker/docs/architecture.md)
Decision log: [`DECISIONS.md`](DECISIONS.md)

## Status

All seven steps implemented. **291 tests passing, ruff and mypy clean.** Pipeline runs end to
end via `car-tracker all`.

| Module | Doc |
|---|---|
| `telemetry.py` | [telemetry.md](car_tracker/docs/telemetry.md) |
| `video.py` | [video.md](car_tracker/docs/video.md) |
| `geo.py` | [geo.md](car_tracker/docs/geo.md) |
| `calibrate.py` | [calibrate.md](car_tracker/docs/calibrate.md) |
| `detect.py` | [detect.md](car_tracker/docs/detect.md) |
| `track.py` | [track.md](car_tracker/docs/track.md) |
| `postprocess.py` | [postprocess.md](car_tracker/docs/postprocess.md) |
| `visualization.py` | [visualization.md](car_tracker/docs/visualization.md) |
| `cli.py` | [cli.md](car_tracker/docs/cli.md) |

Detector decided: YOLO11 + DOTA weights (see D1). Spike in `car_tracker/spikes/`.

## Remaining before submission

| # | Item | Why it matters | Blocker? |
|---|---|---|---|
| 1 | **Two known recall/precision gaps, both measured and documented** | 76 reported vs 56 counted manually. Parking-lot hops inflate it (D7); the 70–85 m detection blind band suppresses it (D8). Both documented rather than fixed | no — decided |
| 2 | ~~Full-video run on the GPU box~~ | ✅ done: 19,698 detections → 286 tracks, 88 moving | — |
| 3 | ~~"Challenges encountered" in README~~ | ✅ done, plus a Known limitations section | — |
| 4 | Commit `results/` artifacts | ✅ `map.html` and `tracks.geojson` are committed | — |
| 5 | ~~Visual verification of `results/map.html`~~ | ✅ done — paths lie on the roads over satellite imagery, validating the geo chain end to end | — |
| 6 | Reapply the gate fixes deliberately | the scalar-max gate is a genuine bug (allowed 25–35 m jumps) and was reverted with everything else. On branch `appearance-experiment` | no |
| 7 | Annotated demo clip | not required; most persuasive evidence detection works | no |
| 8 | `uv.lock` / `requirements-lock.txt` | exact reproducibility | no |

## Assignment deliverables checklist

From the PDF, Task 3:

| Required | Status |
|---|---|
| Code to detect cars, track their paths, plot them on a map | ✅ |
| A map visualization showing the paths of all detected cars | ✅ full video, paths verified on roads; over-reports *moving* cars in parking lots, documented |
| Documentation of approach, methods and assumptions | ✅ README + 9 module docs + architecture |
| Clean, modular code with comments and docstrings | ✅ |
| Instructions on how to run | ✅ README |
| Challenges encountered | ✅ README section + DECISIONS.md D1–D7 |
| GitHub repo or zip with clear directory structure | ✅ committed, `results/` included |

## Known caveats

- **Parking lots over-report moving cars**: 88 vs 56 counted manually. Cars at ~2.5 m pitch
  versus 1.5 m projection error during rapid yaw, so position cannot exclude the
  neighbouring vehicle. Root cause and rejected fixes: DECISIONS.md D7.
- **A real bug is currently present**: the association gate is a single scalar taken as the
  maximum over all tracks, so one coasting track widens the gate for everything — this
  permits 25–35 m single-frame jumps. A per-track fix was written and reverted; see D7.
- **Detection is blind over frames ~3163–4142** (70–85 m altitude): 0.8 detections per
  frame against 4.4–5.8 elsewhere, verified by hand. Cars there are largely missing from
  the map. See D8.
- Reported speeds are now seeded rather than starting from rest, so the old ~30% low bias
  is gone; motion is judged on displacement ÷ duration (D9).
- Detection recall is ~80% per frame, which is fine — a car spans hundreds of frames
  and tracking closes the gaps. Never measured rigorously (2 frames hand-labelled).
- Detector box dimensions are unreliable as vehicle length (D5). Not used anywhere.
- Cross-pass re-identification is out of scope: the flight is out-and-back, so a car
  seen on both passes gets two track IDs.
- Flat-ground assumption, no lens distortion correction, no terrain model.
