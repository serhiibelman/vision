
# postprocess

`car_tracker/src/car_tracker/postprocess.py` — separates moving vehicles from parked
ones and exports the survivors. Implemented. **This is where the assignment is
satisfied.**

## The problem

The detector finds every vehicle, and on this footage most are parked — a snowy street
with roughly twenty stationary cars per frame and a couple driving. A map of all
detections answers the wrong question.

## The discriminator

Geometric, not visual. Once tracks are geo-referenced:

| Signal | Moving car | Parked car |
|---|---|---|
| `displacement_m` (start → end) | 20–32 m observed | ~0 |
| `straightness` (displacement ÷ path length) | ~1.0 | <0.2 |
| `median_speed_mps` | 5–15 | ~0 |

**Straightness** is the strongest signal. A parked car accumulates path length from
noise while going nowhere, so the ratio collapses; a car on a road approaches 1.

## API

```python
from car_tracker.postprocess import moving_tracks, to_geojson, report

moving, features = moving_tracks(tracks)
print(report(features))
to_geojson(moving, features, "results/tracks.geojson")
```

| Member | Does |
|---|---|
| `moving_tracks(tracks, criteria)` | full pass → `(moving observations, all features)` |
| `estimate_gps_drift(tracks)` | common-mode drift per timestamp |
| `remove_gps_drift(tracks)` | subtract it, keeping originals |
| `smooth_positions(tracks, window)` | rolling-median position smoothing |
| `track_features(tracks, ...)` | per-track geometry |
| `classify(features, criteria)` | `is_moving` + `verdict` |
| `to_geojson(tracks, features, path)` | LineString per moving car |
| `report(features)` | counts kept and rejected, by reason |
| `MovingCriteria` | thresholds, frozen dataclass |

## GPS drift correction

Drift in the drone's own fix is **common-mode**: it displaces every projected object
together, so a whole street of parked cars appears to drift in formation and can cross
the moving threshold.

The median step across all tracks present in a frame estimates it. Median rather than
mean, so the handful of genuinely moving vehicles cannot bias the estimate — which holds
as long as most tracks are parked, and here they are.

Corrected values go in new columns (`east_corrected_m`, `north_corrected_m`) so the
correction can be inspected rather than trusted. A test asserts it cancels four parked
cars drifting in formation **and** leaves a genuine mover intact.

## Smoothing

Rolling **median**, not mean: a single badly placed detection is rejected outright
rather than blended into the path.

## Thresholds

| Name | Default | Rejects |
|---|---|---|
| `min_displacement_m` | 15.0 | parked cars, above GPS noise |
| `min_straightness` | 0.4 | tracks that accumulate distance by jittering |
| `min_median_speed_mps` | 1.5 | stationary vehicles |
| `min_duration_s` / `min_observations` | 0.5 / 5 | tracks too brief to judge |
| `max_speed_mps` | 45.0 | association errors masquerading as cars |

## Verdicts, and why order matters

`classify` records the **first failed test** as `verdict`, so a rejection can be argued
with rather than merely dropped:

```
moving · stationary · wandering · too_slow · too_short · implausible_speed
```

Displacement-based tests are checked before speed-based ones. Speed can spike from one
mis-placed detection, so a track that plainly went nowhere should read `stationary`
rather than `implausible_speed`. A genuine association error has large displacement and
falls through to the speed test correctly.

## Rejections stay visible

`moving_tracks` returns features for **every** track, not just survivors, and the map
draws rejected tracks as a hidden layer. If a real car is filtered out, that is where it
shows up.

## Measured on video2

Frames 1000–1300, every 3rd frame:

```
tracks            15
moving            5
rejected: stationary 9
rejected: too_short 1

displacement      20 .. 32 m
median speed      34.6 km/h
duration          1.2 .. 4.0 s
```

All five movers had straightness 1.00. Speeds of 19–54 km/h are plausible for the
street, and displacement ÷ duration independently cross-checks the filtered speed for
four of the five.

## Known caveat

One track (6 observations, 1.2 s) has displacement implying ~78 km/h while its filtered
speed says 35 km/h. Those disagree, so it is either an association error joining two
vehicles or a filter that had not converged. It passes the thresholds; short tracks
deserve scepticism.

## Tests

`car_tracker/tests/test_postprocess.py` — 34 tests. Straightness on driving versus
parked paths, a divide-by-zero guard so a perfectly still car is not "perfectly
straight", drift correction in both directions, median smoothing rejecting an outlier,
GeoJSON structure and lon/lat ordering, and configurable thresholds.
