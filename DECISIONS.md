# Decisions

Running log of choices made during development, and why.

---

## D1 — Detector: YOLO11 with DOTA-trained weights

**Date:** 2026-08-17
**Choice:** `yolo11x-obb.pt` (DOTA), tiled 640×640 at native resolution.

> **PARTLY SUPERSEDED — see D8.** The tiling and never-downscale findings below hold. The
> conclusion "drop COCO" was too strong: it came from aggregate counts over 20 sampled
> frames plus two hand-checked ones. The two weight sets have complementary blind spots,
> and DOTA alone is near-blind over frames ~3163–4142. DOTA-only is still the shipped
> choice, but as an accepted trade rather than because COCO is worse.

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

## D5 — Calibrate the pixel-to-ground transform from the footage

**Date:** 2026-08-17
**Choice:** measure metres-per-pixel and the image→north rotation from data
(GPS displacement ÷ pixel shift). Keep the measurement even though the spec turned
out to be nearly right — see the correction below.

> **CORRECTED 2026-08-17.** This entry originally concluded the spec GSD was ~30%
> wrong, based on measuring car lengths. Calibration showed `gsd_scale = 1.036` —
> the spec is accurate to 3.6%. The car-length method was the flawed part:
> DOTA-OBB boxes are systematically tight (70 px × 4.77 cm/px = **3.34 m** for a real
> ~4.3 m car, so boxes undersize by ~25%). The caveats listed below were the right
> ones; they were simply weighted too lightly against an assumption-free measurement.
>
> **Carry-forward rule: never use detector box dimensions as a length estimate.**

**Camera identified:** MP4 metadata contains `DJI M3T` — Mavic 3 Thermal. Its wide
camera is 1/2" CMOS, f/2.8, **24 mm equivalent**, DFOV 84°. The SRT's `fnum: 2.8`
matches, so `focal_len: 24.00` is confirmed to be the **35 mm equivalent**, not the
actual focal length. That question is closed.

**But the spec-derived GSD is still wrong.** Measured car length against the value
predicted from a 24 mm-equivalent lens on a 1920 px frame:

| Frame | Alt | Car measured | Spec predicts | Measured GSD | Spec GSD | Ratio |
|---|---|---|---|---|---|---|
| 4979 | 61.0 m | 70 px (n=15) | 94 px | 6.42 cm/px | 4.77 | 1.35× |
| 4717 | 60.9 m | 74 px (n=18) | 95 px | 6.06 cm/px | 4.76 | 1.27× |

Cars appear ~25–35% smaller than predicted, so the true field of view is wider than
the still-photo spec implies — most likely because 1080p 16:9 uses a wider crop of
the sensor.

**Caveats on that measurement** (indicative, not exact): car length was assumed to be
4.5 m (real range 4.0–4.8 m), and detector boxes typically undersize by 5–10%.
Neither accounts for a full 1.3× alone, so a genuine discrepancy exists.

**Why it matters:** shipping the spec value would make every distance and speed
~30% wrong.

**Method chosen** — assumption-free, needs no sensor spec, no car length, no box
accuracy:

```
GSD [m/px] = GPS displacement between two frames [m] / static ground shift [px]
```

`telemetry.gsd` is therefore labelled provisional and is superseded by
`calibrate.py`.

**Measured result** — 40 pairs sampled across the flight, 15 frames apart, filtered
to near-constant yaw and altitude:

```
pairs measured   35 of 40        inliers/pair  median 184
gsd_spec         7.91 cm/px      gsd_measured  8.04 cm/px
gsd_scale        1.036           (IQR 1.011 .. 1.059)
yaw convention   rotation = +1 * yaw - 0.39 deg   (spread 0.75 deg)
residual         0.29 m
```

**Yaw convention resolved:** DJI's `gb_yaw` is directly the bearing of image-up.
Sign `+1`, offset −0.39°, consistent to 0.75° across headings from −170° to +175°.
No mirroring.

**Gate passed:** a static ground point projects to within 0.29 m from two different
frames, below GPS noise. Downstream stages are safe to build.

**Why keep calibration now that the spec proved close?** It cost nothing to run, it
is what *proved* the spec correct, and the 0.29 m residual is the accuracy figure the
deliverable needs to report. Trusting the spec without it would have been a guess
that happened to be right.

---

## D6 — Residual measured from tracked features, not a hand-picked landmark

**Date:** 2026-08-17
**Choice:** validate calibration by projecting every RANSAC inlier from both frames of
a pair and measuring how far the two projections disagree.

**Why:** the original plan was to hand-pick a building corner and check that it holds
still. Tracked features are already static ground points by construction — RANSAC
rejected anything moving — so this yields hundreds of samples per pair automatically,
with no manual step and no risk of picking an unlucky landmark.

---

## D7 — Parking-lot false movers: investigated, two fixes tried and reverted

**Date:** 2026-08-18
**Status:** documented limitation. Code is at `0c04272`; the attempts live on branch
`appearance-experiment` if anyone wants to revisit them.

**Trigger:** a manual count of the full video gave 56 moving cars against 88 reported.
Most of the false ones were parked cars in lots, packed tightly together.

### What the data showed (these measurements stand)

- **45 of 88** "moving" tracks travel within 30° of the drone's own heading; **28** also
  match its ground speed to within ±40%. A track that follows the drone is walking along a
  row of parked cars as new ones enter the frame.
- They **cluster in frames 4563–4979**, the final descent over dense parking.
- Some contain **single-frame steps of 25–35 m** — roughly 1000 m/s.
- Projection accuracy is *not* the cause: the static-point residual stays at 0.42 m median
  even at 25–57°/s yaw rate, versus 0.28 m when calm.
- Reported speeds are biased low: displacement ÷ duration over filtered speed has a median
  ratio of 1.45, because the Kalman filter starts at rest and the median includes warm-up.

### Root cause

Cars in a parking lot sit at ~2.5 m pitch. Projection error is 0.3 m typically and 1.5 m
during rapid yaw, plus ~0.5–1 m of centroid jitter. The gate cannot be smaller than our own
error, so the neighbouring car is always reachable, and a parking row is collinear — a hop
chain therefore has straightness ≈ 1 and growing displacement, indistinguishable from a
real car by **position alone**.

### Attempt 1 — fix the gates (a real bug, but not the cure)

Two genuine defects were found and fixed:

1. The gate was a **single scalar taken as the maximum over all tracks**, so one track
   coasting the full `max_coast_s` handed *every* track a 39 m radius. This is what allowed
   the 25–35 m jumps.
2. Even per track, using the global `max_speed` gave a settled parked car a 37 m radius
   after coasting.

Also added: velocity seeded from the first two observations (removing the low-speed bias),
and a `teleporting` verdict rejecting impossible single steps.

**Result:** 35 m jumps eliminated, drone-following tracks down from 28 to 19 — but the
headline count barely moved (88 → 86), because tighter gates **fragment real tracks**
(`too_short` rejections rose from 38 to 504). One error traded for another.

### Attempt 2 — appearance matching (made it worse)

A median BGR colour per detection, with pairs beyond a distance threshold excluded from
association. Reported movers fell 88 → 60, but **the parked chains largely survived while
real moving cars were lost**.

Cause: raw BGR measures illumination, not identity. A car driving from sun into shadow
moves further in BGR than two adjacent parked cars do, so the gate rejected *correct*
matches and fragmented real tracks, while the chains that survived were exactly the
same-colour pairs colour can never separate.

A chromaticity-based descriptor (BGR ÷ intensity, brightness down-weighted) fixes the
shadow problem in isolation — same car through shadow scores 7–15, two differently
coloured cars 170–180 — but it was never validated on the full video before the revert.

### Also tried and rejected

- **Progress consistency** (a hopper dwells then jumps; a real car progresses steadily):
  stalled-window fraction 0.29 for suspects versus 0.13 for plausible tracks. Real
  separation, but thresholding dropped 22 probably-real tracks to remove 11 suspects.
- **Rejecting tracks that move at the drone's speed and heading**: identifies 19 likely
  artifacts, but would also delete a real car driving along the road alongside the drone,
  which is plausible on this footage.

### Untried options

1. **Bound rather than fix**: raise `min_displacement_m` from 15 m to ~40 m. A hop chain
   rarely exceeds a lot's dimension; a real car on a road easily does. One line, immediately
   measurable against the count of 56. Loses genuinely short trips.
2. **Suppress tracks inside static clusters**: find regions of many mutually stationary
   vehicles and apply a stricter test only there, instead of penalising everything.
3. **Accept and report**: ship the map with the limitation documented and the rejected-track
   layer available. This is where the code currently sits.

---

## D8 — Two detectors together: implemented, measured, reverted

**Date:** 2026-08-18
**Choice:** stay with DOTA-trained weights alone.

**Why it was tried.** Detection density collapses over frames ~3163–4142 (70–85 m
altitude) to 0.8 per frame against 4.4–5.8 elsewhere, and the dip is non-monotonic so it is
not an object-size effect. Hand-checked, frame 3620 has ≥9 visible vehicles and one
detection. Confidence is not the cause: dropping the threshold from 0.25 to 0.05 recovers
none of them, and isolated upscaled crops return nothing, so those cars are never proposed.

COCO-trained weights cover much of that band. Measured per frame:

| Frame | DOTA | COCO |
|---|---|---|
| 540 | 8 | 1 |
| 600 | 5 | 3 |
| 660 | 0 | 9 |
| 3570 | 0 | 8 |
| 3620 | 1 | 11 |
| 3710 | 1 | 12 |

Genuinely complementary in both directions — which also **corrects D1**. That entry
concluded "DOTA-only, drop COCO" from aggregate counts over 20 sampled frames plus a close
look at two. Too strong a conclusion from too little evidence: neither weight set covers
this footage alone.

**What was built.** `EnsembleDetector` unioning both models with NMS across the union and a
`source` column recording provenance, plus `filter_by_ground_size` rejecting detections
whose implied ground length cannot be a vehicle. That filter was needed because COCO boxes
buildings; measured, real vehicles run 2.7–5.9 m implied while COCO's false positives reach
12–50 m, so an 8 m bound separated them. Verified: frame 3550's two shed roofs (11.0 m and
11.1 m) were both rejected, frame 3620 went from 1 usable detection to 7.

**Why it was reverted.** The resulting map was not an improvement in review. Doubling
inference (~12 → ~25 min on GPU) and roughly tripling the raw detection count gives
association many more chances to join adjacent parked cars, and parking-lot hops were
already the dominant false-positive source (D7).

**Available on branch `appearance-experiment`'s successor if revisited.** The measurements
above stand regardless; the recall gap over frames 3163–4142 is real and remains
unaddressed, and is documented as a limitation rather than hidden.

---

## D9 — Motion judged by displacement over duration, and by heading consistency

**Date:** 2026-08-18
**Choice:** replace the filtered-speed test with displacement ÷ duration, and add a
heading-spread test. Turn drift correction off by default.

**Filtered speed was unusable as a criterion.** The Kalman filter starts from rest, so
median speed over a track understates by 1.7× typically and up to **49×** on short tracks.
Ten real cars were rejected as `too_slow` while covering 15–39 m with straightness
0.81–0.99 — one travelled 39 m in 2.27 s (62 km/h) and its filtered speed read 1.10 m/s.
Displacement ÷ duration needs no filter convergence. Velocity is now also seeded from the
first two observations so *reported* speeds are honest.

**The speed floor is 5 m/s (18 km/h), not near zero,** because during sustained fast
rotation (~98°/s) parked cars creep 20–28 m over 5–7 s, i.e. ~14 km/h — too slow to be
driving, too fast to be parked. Cost: a vehicle crawling in a jam below 18 km/h is not
reported.

**Heading spread catches what straightness cannot.** Straightness compares endpoints to
path length, so a path that reverses repeatedly but ends far away scores 1.0; one real track
scored 1.0 with a heading spread of 80°. Measured: a straight run scores <10°, a smooth 90°
turn ~26°, a U-turn ~52°, and erratic false movers 60–108°. The limit is 50°, which permits
a U-turn and rejected 28 tracks whose paths visibly could not be driven.

**A short trip counts if its direction is consistent** — 10 m at ≤25° spread — because a car
entering and leaving frame may only be observed for 10–14 m, and the flat 15 m floor was
discarding those.

**Drift correction is off by default.** Its estimator sums a per-frame median whose bias
integrates without bound: 117 m east and −108 m north reported on this footage where the
truth is a few metres. With six moving and six parked cars in a synthetic test it erased
all six real movers. It had also been inert for a long time — `smooth_positions` always read
the raw columns, so the correction was computed and discarded; the flag now does what it
says.

**Net effect on this footage:** 88 → 76 reported moving against a manual count of 56, with
composition improved — 10 real cars recovered, 28 impossible paths removed.

---

## Open

- Static-vs-moving filter thresholds — needs real tracks first.
- Whether a contiguous-segment run changes the detector verdict.
