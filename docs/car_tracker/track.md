
# track

`car_tracker/src/car_tracker/track.py` — links per-frame detections into per-vehicle
paths. Implemented.

## Why metres, not pixels

In pixel space a parked car streaks across the frame at the drone's speed while a car
travelling alongside the drone barely shifts. Pixel distance therefore says nothing
about whether two detections are the same vehicle, and the useful threshold changes
with altitude and drone velocity.

In metres the test is physical: a car cannot cover more than `max_speed × dt`.

Detections are projected to coordinates **before** association, not after.

## API

```python
from car_tracker.track import track_detections, summarise

tracks = track_detections(projected_detections)   # projected by geo.py first
print(summarise(tracks))
```

| Member | Does |
|---|---|
| `track_detections(detections, ...)` | projected detections → track rows |
| `to_local_metres(detections, origin)` | lat/lon → ENU metres, fixed origin |
| `summarise(tracks)` | per-track duration, displacement, path length, speeds |
| `KalmanTrack` | constant-velocity filter for one vehicle |
| `Track` | one track under construction, with lifecycle counters |

## Output

`outputs/tracks.csv`, one row per accepted observation.

| Column | Meaning |
|---|---|
| `track_id` | stable per-vehicle identifier |
| `frame`, `t_sec` | when |
| `lat`, `lon` | filtered position |
| `east_m`, `north_m` | same position in local ENU metres |
| `speed_mps`, `heading_deg` | filter outputs, not differenced positions |
| `conf` | detector confidence of the matched detection |

## Algorithm, per frame

| Step | How |
|---|---|
| Retire | tracks unmatched for longer than `max_coast_s` are closed |
| Predict | constant-velocity Kalman advances each track by `dt` |
| Gate | pairings beyond `max_speed × dt + slack` are rejected |
| Associate | Hungarian assignment (`scipy.optimize.linear_sum_assignment`) |
| Update | matched tracks absorb the measurement |
| Spawn | unmatched detections start new tracks |

Retirement happens **before** association, deliberately. Frames with no detections
never enter the loop, so a purely post-match check let a track coast across an
arbitrarily long gap and then claim a distant detection — a wrong-identity join. A test
covers this.

## Why Hungarian, not greedy

Greedy nearest-neighbour swaps identities when two vehicles pass close to one another.
Hungarian assignment is globally optimal for the frame. Out-of-gate pairs are made
expensive rather than impossible so the solver always has a feasible problem, then
dropped after solving.

## Why a hand-rolled Kalman

The model is four states (`east, north, v_east, v_north`) and the matrices are 4×4, so
a filtering library would add more surface than it saves. Velocity comes out as a
filter output, which matters because the moving/parked decision depends on it — 
differencing noisy positions would be far worse.

## Parameters

| Name | Default | Why |
|---|---|---|
| `max_speed_mps` | 35 | 126 km/h, generous for a city street |
| `gate_slack_m` | 4 | GPS noise + 0.29 m projection residual + centroid jitter |
| `min_hits` | 3 | discards single-frame false positives |
| `max_coast_s` | 1.0 | survives a brief miss; time-based so `--stride` behaves |
| `ACCEL_NOISE_MPS2` | 2.0 | process noise: a car changing speed |
| `MEASUREMENT_NOISE_M` | 2.0 | GPS plus projection error |

## Parked cars are kept

Every detection is tracked, parked cars included. They appear as tracks that jitter in
place under GPS noise. Separating them is
[postprocess](postprocess.md) — deliberately a separate stage, because the decision is
statistical over a whole track rather than per frame.

## Frame spacing matters

Tracking needs frames close enough in time to associate. A run with `--stride 166`
(5.5 s apart) produces zero tracks, correctly: nothing can be linked across that gap.
`--stride 3` at 30 fps gives 0.1 s spacing and works well.

## Measured on video2

Frames 1000–1300, every 3rd frame: 282 detections → 15 tracks.

## Tests

`car_tracker/tests/test_track.py` — 37 tests. Kalman convergence and smoothing, ENU
round-trips, gap bridging versus gap splitting, teleporting detections rejected, and
`test_crossing_cars_keep_distinct_ids`, which two vehicles passing in opposite
directions must survive — greedy matching fails it, Hungarian passes.
