
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

| Signal | Moving car | Parked car / artifact |
|---|---|---|
| `displacement_m` (start → end) | 11–190 m observed | ~0 |
| `straightness` (displacement ÷ path length) | ~0.95 | <0.2 |
| `heading_spread_deg` (direction consistency) | <10° straight, ~26° turning | 60–108° |
| `average_speed_mps` (displacement ÷ duration) | 5–15 | ~0 |

**Two signals do the work, and they catch different things.**

*Straightness* rejects a track that accumulates path length by jittering in place.

*Heading spread* rejects a track that reverses direction repeatedly but still finishes far
away — straightness scores such a path **1.0**, because it only compares endpoints to path
length. One real track scored straightness 1.0 with a heading spread of 80°: that is the
"pointing in impossible directions" failure visible on the map.

**Average speed is displacement ÷ duration, not the Kalman filter's speed.** The filter
starts from rest, so its median understates by 1.7× typically and up to **49×** on short
tracks — enough to reject real cars travelling 60 km/h as "too slow". Ten such cars were
being discarded before this changed.

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

## GPS drift correction — implemented but OFF by default

The idea is sound: drift in the drone's own fix is common-mode, displacing every projected
object together, so a street of parked cars can appear to drift in formation.

The estimator is not. It sums a per-frame median, and the small bias in that median
integrates without bound: on this footage it reports **117 m east and −108 m north** where
the truth is a few metres. Worse, when several vehicles move together the median *becomes*
the traffic's motion, so subtracting it erases the real movers — in a synthetic test with
six moving and six parked cars it removed all six real ones.

Enable with `moving_tracks(tracks, correct_drift=True)` only with evidence of real drift,
and check `estimate_gps_drift()` output before trusting it.

It also had no effect for a long time: `smooth_positions` always read the raw columns, so
the correction was computed and silently discarded. That is fixed — the flag now does what
it says.

## Smoothing

Rolling **median**, not mean: a single badly placed detection is rejected outright
rather than blended into the path.

## Thresholds

| Name | Default | Rejects |
|---|---|---|
| `min_displacement_m` | 15.0 | parked cars, above GPS noise |
| `short_trip_displacement_m` / `short_trip_spread_deg` | 10.0 / 25° | concession: a briefly-seen car counts at 10 m **if** its direction is consistent |
| `max_heading_spread_deg` | 50° | paths that reverse and wander; 50 permits a U-turn (~52°) and junction turns (~48°) |
| `min_straightness` | 0.4 | tracks that accumulate distance by jittering |
| `min_average_speed_mps` | 5.0 | stationary vehicles, and rotation artifacts that creep at ~14 km/h |
| `min_duration_s` / `min_observations` | 0.5 / 5 | tracks too brief to judge |
| `max_speed_mps` | 45.0 | association errors masquerading as cars |

## Verdicts, and why order matters

`classify` records the **first failed test** as `verdict`, so a rejection can be argued
with rather than merely dropped:

```
moving · too_short · stationary · wandering · erratic · too_slow · implausible_speed
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

Full video, every frame:

```
tracks                313
moving                 76      <- manual count of the footage: 56
rejected: stationary  124
rejected: too_short    76
rejected: erratic      28
rejected: implausible_speed 5
rejected: too_slow      3
rejected: wandering     1

displacement       11 .. 190 m
average speed      46 km/h
duration          0.5 .. 14.2 s
```

## Known caveats: 76 reported versus 56 actual

Two independent errors, pulling in opposite directions.

**Over-reporting — parking lots.**

The over-count comes from **dense parking lots**. Cars sit at ~2.5 m pitch while projection
error reaches 1.5 m during rapid yaw, so the association gate cannot exclude the
neighbouring vehicle. A parking row is collinear, so a hop chain has straightness ≈ 1 and
growing displacement — indistinguishable from a real car by position alone.

Evidence from the full run: 45 of 88 movers travel within 30° of the drone's own heading,
28 also match its speed, and they cluster in frames 4563–4979 over dense parking.

Two mitigations were tried and reverted (tighter gates fragmented real tracks; raw-BGR
colour matching lost real movers because it measures illumination rather than identity).
Measurements and untried options: DECISIONS.md D7.

**Under-reporting — the detection blind band.** Over frames ~3163–4142 (70–85 m altitude)
the detector averages 0.8 detections per frame against 4.4–5.8 elsewhere, verified by hand
on frame 3620: nine visible vehicles, one detected. Nothing this module can do — the cars
never reach it. See [detect.md](detect.md) and DECISIONS.md D8.

So the count is not simply 20 too high: it is inflated by parking-lot hops and deflated by
missing detections, and the two do not cancel in any principled way. Judging a threshold
change by the headline number alone is therefore misleading — that mistake cost several
iterations.

## Tests

`car_tracker/tests/test_postprocess.py` — 53 tests. Straightness on driving versus parked
paths, a divide-by-zero guard so a perfectly still car is not "perfectly straight", drift
correction in both directions **and** that enabling it reaches the measurement at all,
median smoothing rejecting an outlier, GeoJSON structure and lon/lat ordering, and
configurable thresholds.

Two cases pin the heading test specifically: a **stuttering** path (forward-back-forward,
straightness 0.6 so that test is fooled, spread 180° so heading catches it) must be
rejected, and a **sharp junction turn** (spread 48°) must survive. The short-trip concession
has matching pairs: 12 m travelled straight counts, 12 m travelled jittery does not.
