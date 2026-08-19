
# visualization

`car_tracker/src/car_tracker/visualization.py` — renders maps. Implemented.

## API

```python
from car_tracker.visualization import render_tracks, render_flight_path

render_flight_path(telemetry, "outputs/flight_path.html")
render_tracks(moving, features, "results/map.html",
              telemetry=segment, all_tracks=tracks)
```

| Function | Does |
|---|---|
| `render_tracks(tracks, features, out, telemetry, all_tracks)` | **the deliverable map** |
| `render_flight_path(telemetry, out_path, step=10)` | drone track only |
| `base_map(lat, lon, zoom=None)` | empty map, satellite + street layers |
| `add_car_tracks(fmap, tracks, features)` | car paths onto an existing map |
| `add_rejected_tracks(fmap, tracks, features)` | parked tracks, hidden layer |
| `add_flight_path(fmap, telemetry, step=10)` | drone track onto an existing map |
| `path_length_m(lat, lon)` | ground distance along a polyline, metres |

## Output

`results/map.html` — 48 KB, self-contained, opens in any browser.

| Layer | Content |
|---|---|
| Moving cars | one coloured polyline per car; dot = start, triangle = end |
| Rejected (parked) | grey dashed, **hidden by default**, tooltip names the reason |
| Drone flight path | red line, altitude/yaw/GSD probe tooltips |
| Satellite / Street | basemap toggle |

Car tooltips carry speed in km/h, displacement in metres and duration.

`outputs/flight_path.html` (28 KB) is the flight path alone, from
`render_flight_path`.

Satellite is the default base layer because roads and vehicles must be visible; OSM's
rendering hides both under labels.

Rejected tracks are drawn on purpose: if a genuine car was filtered out by
[postprocess](postprocess.md), that layer is where it shows up. Start and end markers
differ in shape because direction of travel is otherwise ambiguous on a bare line.

## Notes

- `step=10` decimates the polyline. 4979 vertices bloat the HTML and render
  identically at this zoom.
- `path_length_m` uses `pyproj.Geod` on WGS84, not degree arithmetic.

## Measured on video2.SRT

```
path length   2943 m over 166 s  ->  mean 17.7 m/s (64 km/h)
route         out-and-back: west to lon 25.9017, then back east to 25.9166
turnaround    ~frame 2491 (t=83 s), yaw flips -97deg -> +80deg
altitude      101 m held until ~frame 3487, then steps down to 61 m
```

**Consequence of the out-and-back route:** the same parked cars are filmed twice,
from opposite headings. Useful for validating the static filter — a parked car must
land on the same coordinates on both passes. Also means a moving car seen on both
passes gets two track IDs, since cross-pass re-identification is out of scope.

## Purpose in the pipeline

Cheapest possible check on telemetry before the calibration gate:

| Check | Catches |
|---|---|
| path lies on real streets | lat/lon swapped, wrong hemisphere |
| line is smooth | GPS jumps, dropped fixes |
| length is plausible | unit or scale error |

Also used to pick the static landmark for step 3 calibration.

## Tests

`car_tracker/tests/test_visualization.py` — 19 tests, including car-track rendering with
speed labels and the rejected layer. Geodesic distance against a known
1-degree baseline, layer presence, HTML output, decimation, single-row edge case.
