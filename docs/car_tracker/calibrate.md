
# calibrate

`car_tracker/src/car_tracker/calibrate.py` — measures the pixel-to-ground transform
from the footage. Implemented. **This is the pipeline gate.**

## Idea

Between two frames the drone travels a known distance and bearing per GPS, while the
static ground slides across the image by a measurable number of pixels:

```
GSD      = GPS distance / pixel shift magnitude
rotation = GPS bearing - direction of camera motion within the image
```

One observation, both unknowns. No sensor spec, no object sizes, no box accuracy.

## API

```python
from car_tracker.calibrate import calibrate

result = calibrate("video2.MP4", telemetry, image_size=(1920, 1080), count=40)
print(result.report())
model = CameraModel(calibration=result.calibration)
```

| Function | Does |
|---|---|
| `calibrate(video, telemetry, ...)` | full run → `CalibrationResult` |
| `select_pairs(telemetry, ...)` | choose usable frame pairs |
| `measure_shift(image_a, image_b)` | ground motion between two frames |
| `measure_pairs(video, pairs, size)` | measure every pair → DataFrame |
| `fit_yaw_convention(measurements)` | `(sign, offset, spread)` |
| `static_point_residual_m(...)` | accuracy check |
| `wrap180`, `circular_mean_deg` | angle helpers |

## How a pair is measured

1. `goodFeaturesToTrack` on frame A (up to 2000 corners; winter fields are low-texture)
2. `calcOpticalFlowPyrLK` to follow them into frame B
3. `estimateAffinePartial2D` with RANSAC → similarity transform
4. Shift is read at the **image centre**, not from the translation column — with any
   rotation present those differ
5. Camera motion is the opposite of the ground motion, expressed as an angle
   clockwise from image-up

**Moving cars need no masking.** They fail to fit the dominant ground motion and
RANSAC discards them as outliers. There is a test for exactly this.

## Pair selection

A pair is usable when the drone translates enough to measure precisely while the
geometry barely changes, so image motion is near-pure translation at constant scale.

| Filter | Default | Reason |
|---|---|---|
| gap between frames | 15 (~0.5 s) | long enough for a large shift |
| GPS distance | ≥ 3 m | short baselines are noisy |
| yaw change | ≤ 1° | keeps motion near-pure translation |
| altitude change | ≤ 0.5 m | keeps scale constant across the pair |
| RANSAC inliers | ≥ 30 | otherwise the pair is rejected |

Pairs are spread evenly over the flight so no single stretch of terrain dominates and
the yaw fit sees many headings.

Failed pairs are recorded with a reason rather than aborting the run — low-texture
frames over snow are expected.

## Accuracy check

Every RANSAC inlier is a static ground point, so projecting one from both frames of a
pair must give the same coordinate. The distance between the two projections is the
residual.

This replaced hand-picking a landmark: hundreds of samples per pair, fully automatic,
no risk of choosing an unlucky point.

## Measured on video2

```
pairs measured   35 of 40        inliers/pair  median 184
shift            median 118.7 px
gsd_spec         7.91 cm/px      gsd_measured  8.04 cm/px
gsd_scale        1.036           (IQR 1.011 .. 1.059)
yaw convention   rotation = +1 * yaw - 0.39 deg   (spread 0.75 deg)
residual         0.29 m          <- GATE PASSED
```

**Yaw convention:** DJI's `gb_yaw` is directly the bearing of image-up. Sign `+1`,
offset −0.39°, consistent to 0.75° across headings from −170° to +175°. No mirroring.

**GSD:** the camera spec was accurate to 3.6%. An earlier estimate from car lengths
suggested ~30% error and was wrong — DOTA-OBB boxes undersize cars by ~25%. See
DECISIONS.md D5. Never treat box dimensions as a length estimate.

## Median, not mean

`gsd_scale` is the median across pairs. A single mistracked pair would otherwise drag
the scale with it, and one bad scale corrupts every distance downstream.

## Tests

`car_tracker/tests/test_calibrate.py` — 30 tests on synthetic transforms with known
translation, rotation and scale, plus angle-wrapping edge cases, pair-selection
filters, and yaw-convention recovery for both signs and several offsets.
