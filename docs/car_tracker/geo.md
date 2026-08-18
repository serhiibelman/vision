
# geo

`car_tracker/src/car_tracker/geo.py` — converts image pixels to geographic
coordinates. Implemented.

## Why it is simple

The gimbal points straight down in all 4979 frames, so pixel→ground is a similarity
transform, not a perspective projection:

```
1. offset the pixel from the image centre
2. scale by GSD                -> metres
3. rotate so image axes face north
4. add the drone's own position
```

## API

```python
from car_tracker.geo import CameraModel, Calibration

model = CameraModel(image_size=(1920, 1080),
                    calibration=Calibration(gsd_scale=1.036, yaw_offset_deg=-0.39))

lat, lon = model.pixel_to_latlon(u, v, drone_lat, drone_lon, gsd, yaw_deg)
projected = model.project_detections(detections, telemetry)   # whole table at once
```

| Member | Does |
|---|---|
| `CameraModel.pixel_to_latlon(...)` | pixel → coordinate; broadcasts over arrays |
| `CameraModel.project_detections(det, telemetry)` | joins on `frame`, projects in one pass |
| `CameraModel.pixel_offset_metres(...)` | east/north metres from image centre |
| `CameraModel.rotation_deg(yaw)` | bearing of image-up under the calibration |
| `CameraModel.footprint_m(gsd)` | ground size covered by a frame |
| `CameraModel.centre` | principal point, assumed image centre |
| `metres_to_latlon` / `latlon_to_metres` | east/north ↔ coordinates |
| `haversine_m`, `bearing_deg` | ground distance, forward azimuth |

## Calibration

`Calibration` is a frozen dataclass holding the measured transform parameters.

| Field | Meaning | Measured value |
|---|---|---|
| `gsd_scale` | multiplier on the spec GSD | 1.036 |
| `yaw_offset_deg` | constant added to telemetry yaw | −0.39 |
| `yaw_sign` | +1 or −1 | +1 |
| `residual_m` | static-point drift; the accuracy figure | 0.29 |

Defaults are identity (`scale=1.0, offset=0.0, sign=+1`) — uncalibrated. Values come
from `calibrate.py`, never from the camera spec.

## Conventions

- `rotation_deg` = bearing of image-up, clockwise from true north.
  `rotation = yaw_sign * yaw + yaw_offset`.
- Image `v` grows downward, north grows upward, so the vertical axis is negated
  before rotation.
- Rotation is about the image centre, so the centre pixel always maps to the drone's
  own position regardless of yaw.

## One Earth model

All conversions go through `pyproj` geodesics on WGS84 — offset, distance and
bearing alike.

An earlier version used a spherical radius for `metres_to_latlon` while distances
used the ellipsoid. That 0.1% mismatch (~1 cm per 8 m) would have put a floor under
the calibration residual that no amount of fitting could get below. A test comparing
a projected offset against `haversine_m` caught it.

## Verified externally

Beyond the 0.29 m calibration residual, which is a self-consistency check, the projected
car paths were confirmed by eye to lie on the actual roads when drawn over satellite
imagery. A sign or scale error would show up immediately as paths crossing buildings or
fields.

## Assumptions

- Flat ground at the projection plane — no DEM, so terrain relief adds error
- Principal point at image centre, no lens distortion correction (no calibration data
  for this camera)
- GPS accurate to a few metres, so relative path shape is better than absolute
  position

## Tests

`car_tracker/tests/test_geo.py` — 46 tests. Centre pixel invariance under yaw,
cardinal-direction mapping, offset distance equals pixels × GSD, `gsd_scale`
proportionality, metre round-trips, and one asserting a `yaw_sign` flip mirrors the
output — the exact failure the calibration gate exists to catch.
