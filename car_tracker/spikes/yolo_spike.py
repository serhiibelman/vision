"""Spike: does COCO-pretrained YOLO detect cars in nadir drone footage?

This is a throwaway experiment, deliberately kept outside the ``car_tracker``
package. It exists to answer one go/no-go question before the real detection stage
is written, and it is retained afterwards as evidence of *how* the detector was
chosen.

The question matters because COCO was trained on ground-level photography, while
this footage looks straight down (gimbal pitch is a constant -90 degrees) at
vehicles occupying only ~56-94 px. Three configurations are compared:

    A  yolo11x, 640x640 tiles at native resolution   -> cars stay 56-94 px
    B  yolo11x, full frame at imgsz=1920             -> cars stay 56-94 px
    C  yolo11n, full frame at imgsz=640  (naive)     -> cars shrink to ~19-31 px

Config C is included precisely because it is expected to fail: at the library's
default ``imgsz=640`` a 1920-wide frame is downscaled 3x, and demonstrating that
the failure is a resolution mistake rather than a model limitation is part of the
result.

Detection runs at a deliberately low confidence (0.15, below the 0.25 intended for
production) to measure what the model *can* find. A threshold can always be raised
later; detections never made cannot be recovered.

Usage:
    python spikes/yolo_spike.py [--frames N] [--configs A,B,C]

Outputs:
    outputs/spike/<config>/frame_XXXXX.jpg   annotated frames for visual inspection
    outputs/spike/summary.json               per-frame detection counts
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Repo layout: <root>/car_tracker/spikes/yolo_spike.py
ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "tech-assignment" / "video2.MP4"
SRT = ROOT / "tech-assignment" / "video2.SRT"
OUT = ROOT / "car_tracker" / "outputs" / "spike"

# COCO class ids kept: car, bus, truck. Motorcycle (3) is excluded — the assignment
# asks for cars, and at this altitude a motorcycle is a handful of pixels.
VEHICLE_CLASSES = [2, 5, 7]

# DOTA class ids for the -obb weights: 9 = large vehicle, 10 = small vehicle.
# DOTA is aerial/satellite imagery, so unlike COCO it actually contains nadir views.
DOTA_VEHICLE_CLASSES = [9, 10]

CONF = 0.15
TILE, STRIDE = 640, 512

# Provisional only, for the report table. `focal_len: 24.00` in the SRT is assumed
# to be a 35 mm equivalent on a 36 mm-wide frame. The real pipeline replaces this
# with flow-vs-GPS self-calibration (calibrate.py) rather than trusting the spec.
SENSOR_W_EQUIV_MM, FOCAL_EQUIV_MM = 36.0, 24.0
CAR_LENGTH_M = 4.5


@dataclass
class Config:
    """One detector configuration under test."""

    key: str
    weights: str
    tiled: bool
    imgsz: int
    note: str
    obb: bool = False
    classes: list[int] = field(default_factory=lambda: list(VEHICLE_CLASSES))


CONFIGS = {
    "A": Config("A", "yolo11x.pt", True, TILE, "COCO tiled 640, native res"),
    "B": Config("B", "yolo11x.pt", False, 1920, "COCO full frame, imgsz=1920"),
    "C": Config("C", "yolo11n.pt", False, 640, "COCO naive default, imgsz=640"),
    "D": Config(
        "D", "yolo11x-obb.pt", True, TILE, "DOTA-OBB tiled 640",
        obb=True, classes=DOTA_VEHICLE_CLASSES,
    ),
}


@dataclass
class FrameInfo:
    """A sampled frame plus the telemetry-derived scale expected at that moment."""

    index: int  # 1-based, matching the SRT's FrameCnt
    rel_alt: float
    gsd_m_per_px: float
    car_px: float
    counts: dict[str, int] = field(default_factory=dict)
    mean_conf: dict[str, float] = field(default_factory=dict)


def parse_rel_alt(srt_path: Path) -> dict[int, float]:
    """Map 1-based frame number -> relative altitude in metres.

    A minimal stand-in for the real telemetry parser, which does not exist yet.
    ``FrameCnt`` and ``rel_alt`` each occur exactly once per SRT block and in the
    same order, so zipping the two match lists is sufficient here.
    """
    text = srt_path.read_text(encoding="utf-8", errors="replace")
    frames = [int(v) for v in re.findall(r"FrameCnt:\s*(\d+)", text)]
    alts = [float(v) for v in re.findall(r"rel_alt:\s*([-\d.]+)", text)]
    if len(frames) != len(alts):
        raise ValueError(f"SRT mismatch: {len(frames)} FrameCnt vs {len(alts)} rel_alt")
    return dict(zip(frames, alts, strict=True))


def gsd_for_altitude(alt_m: float, image_width_px: int) -> float:
    """Ground sample distance (metres per pixel) for a nadir camera at ``alt_m``."""
    half_fov = np.arctan(SENSOR_W_EQUIV_MM / (2 * FOCAL_EQUIV_MM))
    ground_width_m = 2 * alt_m * np.tan(half_fov)
    return ground_width_m / image_width_px


def tile_origins(width: int, height: int, tile: int, stride: int) -> list[tuple[int, int]]:
    """Top-left corners covering the frame, with the final row/column clamped inward.

    Clamping (rather than padding) keeps every tile full-size, so the model never
    sees black borders, at the cost of slightly more overlap at the edges.
    """

    def axis(total: int) -> list[int]:
        if total <= tile:
            return [0]
        origins = list(range(0, total - tile + 1, stride))
        if origins[-1] != total - tile:
            origins.append(total - tile)
        return origins

    return [(x, y) for y in axis(height) for x in axis(width)]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> list[int]:
    """Plain greedy non-maximum suppression over xyxy boxes.

    Hand-rolled to keep the spike free of a torchvision import; the merge across
    overlapping tiles is the only place suppression is needed.
    """
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]
    return keep


def _extract(res, obb: bool) -> tuple[np.ndarray, np.ndarray]:
    """Pull axis-aligned xyxy boxes and confidences out of a Results object.

    Oriented-box models expose ``res.obb`` instead of ``res.boxes``. The oriented
    box's own axis-aligned envelope (``.xyxy``) is used here so both model families
    are measured on the same footing; the rotation in ``.xywhr`` is what makes these
    weights interesting later, since it yields vehicle heading directly.
    """
    container = res.obb if obb else res.boxes
    if container is None or not len(container):
        return np.empty((0, 4)), np.empty(0)
    return container.xyxy.cpu().numpy(), container.conf.cpu().numpy()


def detect(model, frame: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Run one configuration on one frame; return (xyxy boxes, confidences)."""
    height, width = frame.shape[:2]

    if not cfg.tiled:
        res = model(frame, imgsz=cfg.imgsz, conf=CONF, classes=cfg.classes, verbose=False)[0]
        return _extract(res, cfg.obb)

    origins = tile_origins(width, height, TILE, STRIDE)
    crops = [frame[y : y + TILE, x : x + TILE] for x, y in origins]
    results = model(crops, imgsz=TILE, conf=CONF, classes=cfg.classes, verbose=False)

    boxes, confs = [], []
    for (ox, oy), res in zip(origins, results, strict=True):
        xyxy, conf = _extract(res, cfg.obb)
        if not len(xyxy):
            continue
        boxes.append(xyxy + np.array([ox, oy, ox, oy]))  # tile -> frame coords
        confs.append(conf)

    if not boxes:
        return np.empty((0, 4)), np.empty(0)

    boxes_arr, confs_arr = np.vstack(boxes), np.concatenate(confs)
    keep = nms(boxes_arr, confs_arr)
    return boxes_arr[keep], confs_arr[keep]


def annotate(
    frame: np.ndarray, boxes: np.ndarray, confs: np.ndarray, caption: str
) -> np.ndarray:
    """Draw detections and a caption banner onto a copy of the frame."""
    canvas = frame.copy()
    for (x1, y1, x2, y2), conf in zip(boxes, confs, strict=True):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(canvas, p1, p2, (0, 255, 0), 2)
        cv2.putText(
            canvas, f"{conf:.2f}", (p1[0], max(12, p1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA,
        )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        canvas, caption, (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return canvas


def pick_device() -> str | int:
    """CUDA, else Apple MPS, else CPU. Never hardcoded — the repo runs on both."""
    import torch

    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=20, help="frames sampled across the flight")
    ap.add_argument("--configs", default="A,B,C", help="comma-separated config keys")
    args = ap.parse_args()

    if not VIDEO.exists():
        raise SystemExit(f"video not found: {VIDEO}")

    from ultralytics import YOLO

    device = pick_device()
    alts = parse_rel_alt(SRT)
    cap = cv2.VideoCapture(str(VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Spread samples across the whole flight so the 102 m -> 61 m altitude change
    # acts as a free ablation on apparent object size.
    indices = np.linspace(1, total, args.frames, dtype=int)

    print(f"video   {VIDEO.name}  {width}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
          f"@ {fps:.3f} fps, {total} frames")
    print(f"device  {device}")
    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    print(f"conf    {CONF} (deliberately low)")
    for k in keys:
        print(f"        [{k}] {CONFIGS[k].note:<28} classes {CONFIGS[k].classes}")
    print()

    models = {k: YOLO(CONFIGS[k].weights) for k in keys}
    for k in keys:
        OUT.joinpath(k).mkdir(parents=True, exist_ok=True)

    infos: list[FrameInfo] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx) - 1)  # SRT is 1-based, OpenCV 0-based
        ok, frame = cap.read()
        if not ok:
            print(f"frame {idx}: read failed, skipping")
            continue

        alt = alts.get(int(idx), float("nan"))
        gsd = gsd_for_altitude(alt, width)
        info = FrameInfo(int(idx), alt, gsd, CAR_LENGTH_M / gsd)

        for k in keys:
            cfg = CONFIGS[k]
            boxes, confs = detect(models[k], frame, cfg)
            info.counts[k] = len(boxes)
            info.mean_conf[k] = float(confs.mean()) if len(confs) else 0.0
            caption = (
                f"[{k}] {cfg.note} | frame {idx} | alt {alt:.1f}m | "
                f"gsd {gsd * 100:.1f}cm/px | car~{info.car_px:.0f}px | {len(boxes)} det"
            )
            cv2.imwrite(
                str(OUT / k / f"frame_{idx:05d}.jpg"),
                annotate(frame, boxes, confs, caption),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )

        counts = "  ".join(f"{k}={info.counts[k]:>3}" for k in keys)
        print(f"frame {idx:>5}  alt {alt:>6.1f}m  gsd {gsd * 100:>4.1f}cm/px  "
              f"car~{info.car_px:>3.0f}px   {counts}")
        infos.append(info)

    cap.release()

    print("\n--- totals " + "-" * 50)
    for k in keys:
        total_det = sum(i.counts[k] for i in infos)
        nonzero = sum(1 for i in infos if i.counts[k] > 0)
        mean_conf = np.mean([i.mean_conf[k] for i in infos if i.counts[k]] or [0])
        print(f"[{k}] {CONFIGS[k].note:<26} {total_det:>5} detections   "
              f"frames with >=1: {nonzero}/{len(infos)}   mean conf {mean_conf:.2f}")

    summary = {
        "video": str(VIDEO),
        "fps": fps,
        "conf": CONF,
        "classes": VEHICLE_CLASSES,
        "device": str(device),
        "configs": {k: CONFIGS[k].note for k in keys},
        "frames": [
            {
                "index": i.index,
                "rel_alt_m": i.rel_alt,
                "gsd_cm_per_px": i.gsd_m_per_px * 100,
                "car_px": i.car_px,
                "counts": i.counts,
                "mean_conf": i.mean_conf,
            }
            for i in infos
        ],
    }
    OUT.joinpath("summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nannotated frames -> {OUT}")
    print("Now LOOK at them: do the cars driving on the road have boxes?")


if __name__ == "__main__":
    main()
