
# detect

`car_tracker/src/car_tracker/detect.py` — finds vehicles in each frame. Implemented.

## API

```python
from car_tracker.detect import VehicleDetector, frame_range
from car_tracker.video import probe

info = probe("video2.MP4")
detector = VehicleDetector()                       # yolo11x-obb.pt, conf 0.25
detections = detector.detect_video(
    "video2.MP4",
    frame_range(info.frame_count, stride=1),
    out_path="outputs/detections.csv",
)
```

| Member | Does |
|---|---|
| `VehicleDetector.detect_frame(image, frame)` | one frame → `FrameDetections` |
| `VehicleDetector.detect_video(path, frames, out_path)` | many frames → DataFrame + CSV |
| `tile_origins(w, h, tile, stride)` | tile corners covering the frame |
| `nms(boxes, scores)` | de-duplicate across tile overlaps |
| `frame_range(count, stride, limit)` | 1-based frame numbers to process |
| `pick_device()` | CUDA → MPS → CPU |

## Output

`outputs/detections.csv`, one row per detection.

| Column | Meaning |
|---|---|
| `frame` | 1-based frame number |
| `u`, `v` | oriented-box centroid, pixels — consumed by `geo.project_detections` |
| `conf` | detection score |
| `cls` | DOTA class: 9 large vehicle, 10 small vehicle |
| `w_px`, `h_px` | box size — **diagnostic only**, see below |
| `angle_deg` | box rotation, feeds the moving/parked filter |

Written incrementally and flushed per frame, so a run that dies at frame 4800 keeps
everything before it.

## How a frame is processed

1. Split into 640×640 tiles, stride 512 (~20% overlap) → **8 tiles** for 1920×1080
2. All 8 tiles in **one batched model call**
3. Shift tile-local boxes into full-frame coordinates
4. NMS on the axis-aligned envelopes to drop cars seen twice across a seam

Tiles are clamped inward at the last row/column rather than padded, so the model never
sees black borders.

## Why tiled at native resolution

At the library default `imgsz=640` a 1920-wide frame is downscaled 3×, shrinking a car
from ~57 px to ~19 px. Across a 20-frame sample that collapsed detections from 61 to
4. Tiling keeps cars at native size. See DECISIONS.md D1.

## Why confidence is low (0.25)

Temporal consistency filters false positives better than a high threshold: a real car
appears in hundreds of consecutive frames, a spurious box usually does not. Detections
never made cannot be recovered, so the threshold stays permissive and the minimum
track length does the filtering later.

## Measured on video2 (CPU laptop, 30 frames sampled)

```
124 detections over 30 frames, 25 frames with >=1
confidence  mean 0.58, range 0.25 .. 0.87
classes     63 large vehicle, 61 small vehicle
runtime     11.6 s/frame on 4 CPU cores
```

| Machine | Full 4979 frames |
|---|---|
| This laptop (CPU) | **~16 h** — not viable |
| NVIDIA GPU box | ~10–15 min |

So develop and test here, run the full pass on the GPU box. `--stride` exists for
partial CPU runs.

## Measured limitation: a blind altitude band

Detection density across the flight is **not** uniform, and not a function of object size:

| Altitude band | Frames | Detections per frame |
|---|---|---|
| ≤70 m | ~4200–4979 | 5.8 |
| **70–85 m** | **~3163–4142** | **0.8 (median 0)** |
| 85–105 m | 1–3160 | 4.4 |

The dip is non-monotonic — recall collapses in one band and recovers on both sides — so it
cannot be explained by cars being too small or too large. Camera settings are identical
across the bands (ISO 100, ~1/1100 s), and the drone passes over the *same* streets on the
way out and back, so the scene content is comparable.

Verified by hand on frame 3620 (77 m): at least **nine** clearly visible vehicles, one
detected. Lowering confidence does not recover them —

| conf | 0.25 | 0.15 | 0.10 | 0.05 |
|---|---|---|---|---|
| detections on frame 3620 | 1 | 1 | 1 | 1 |

— so the cars are not scoring low, they are not being proposed at all. Isolated 640 px
crops around individual missed cars, even upscaled 2×, also return nothing.

**Consequence:** moving-car recall is materially worse over roughly frames 3163–4142.
Judged from the detection funnel alone this is invisible — 99% of detections survive
tracking — because the loss happens before anything is detected.

COCO-trained weights cover much of this band (frame 3620: 11 detections versus 1) but miss
what DOTA finds elsewhere and box buildings. Running both was implemented, measured and
then reverted; see DECISIONS.md D8.

## Do not use `w_px` / `h_px` as vehicle length

Box dimensions are unreliable and inconsistent across altitude: at ~102 m they imply
~4.8 m vehicles, at ~61 m ~3.4 m for the same kind of car. The class split (roughly
half "large vehicle") is likewise not trustworthy at ~57 px. Scale comes from
`calibrate.py`, which is independent of any box.

## Tests

`car_tracker/tests/test_detect.py` — 39 tests with the model stubbed, so no weights and
no GPU are needed. Covers full-frame tile coverage, clamping, tile→frame coordinate
shift, NMS de-duplication of a car straddling a seam, angle conversion, incremental
CSV writing, and one test asserting the output drops straight into
`geo.project_detections`.
