# Decisions

Running log of choices made during development, and why.

---

## D1 — Detector: YOLO11 with DOTA-trained weights

**Date:** 2026-08-17
**Choice:** `yolo11x-obb.pt` (DOTA), tiled 640×640 at native resolution.

**Why:** measured on 20 frames sampled across the video, hand-verified on 2 frames.

| Config | Weights | Input | Detections | Result |
|---|---|---|---|---|
| A | COCO | 640 tiles, native res | 61 | ~29% recall, detects buildings as cars |
| B | COCO | full frame @1920 | 33 | worse |
| C | COCO | full frame @640 (default) | 4 | unusable |
| **D** | **DOTA** | **640 tiles, native res** | **78** | **~80% recall, 0 false positives** |

Two variables isolated: A vs C = resolution only; A vs D = training data only.

- COCO is ground-level photography — it has effectively never seen a car from
  above, so snowy roofs score as vehicles (a whole building at 0.42 confidence).
- DOTA is aerial/satellite imagery — the correct domain.
- Never downscale: at the library default `imgsz=640` a 1920-wide frame shrinks
  cars from ~57 px to ~19 px. That alone drops detections from 61 to 4.
- `-obb` gives oriented boxes, so each detection carries a heading — useful for
  the moving/parked filter later.

**Rejected alternatives:** RT-DETR, Faster R-CNN, DETR — all ship COCO-trained and
would hit the same nadir domain gap. Architecture matters less than training data.
Fine-tuning our own model: hours of GPU time for no gain over pretrained DOTA.
Motion-based detection (frame differencing / optical flow residual): finds any
motion, cannot assert "car" — flags shadows, trees and roof parallax.

**Confidence:** high on the direction, low on the exact numbers (only 2 frames
hand-labelled, 0.4% of frames processed, no contiguous segment tested). Cheap to
reverse — swapping weights is one line.

**Not yet verified:** performance on *moving* cars. No contiguous frames were run,
so motion blur on moving vehicles is untested.

---

## D2 — Per-frame recall does not need to be high

**Choice:** accept ~80% per-frame recall; do not chase 99%.

**Why:** a car visible for 10 s appears in ~300 frames. Missing 20% still leaves
~240 detections. Gaps are closed by tracking (Kalman interpolation). Per-*car*
recall across a track approaches ~99% even at 80% per-frame.

The real problem is the opposite: the detector reliably finds every **parked** car
too, and this footage is a street with ~20 parked cars per frame. Filtering those
out is the actual task.

---

## D3 — Telemetry drives geo-referencing

**Choice:** use the SRT sidecar (one block per video frame) to convert pixels to
latitude/longitude.

**Why:** without it, car paths exist only in pixels inside a frame that is itself
moving — impossible to place on a map. The SRT provides per-frame camera position,
altitude (→ metres per pixel), and gimbal yaw (→ image orientation).

`gb_pitch` is constant at −89.9/−90.0 for all 4979 frames, so the camera is pure
nadir and pixel→ground is a simple scale + rotation rather than full perspective
geometry.

---

## D4 — Track in metres, not pixels

**Choice:** georeference detections *before* associating them across frames.

**Why:** in pixel space a parked car streaks across the frame at the drone's speed,
while a car driving alongside the drone barely moves. Pixel distance is therefore
meaningless as an association gate, and it changes with altitude and drone velocity.
In metres the gate is physical: a car cannot move more than ~3 m in 0.1 s.

---

## Open

- Static-vs-moving filter thresholds — needs real tracks first.
- Whether a contiguous-segment run changes the detector verdict.
