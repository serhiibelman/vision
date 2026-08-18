"""
Manual-inspection helper for the detector spike.
11111111111111111111111111111111111111111111111111111111
Two ways to eyeball which model found which car:

``--gallery``
    Build an HTML page pairing the already-rendered annotated frames side by side,
    so the configs can be scrolled and zoomed in a browser. Instant — it only
    references existing JPEGs, no inference.

``--overlay 1049,4979``
    Re-run two models on the named frames and draw both onto a *single* image:
    COCO in red, DOTA-OBB in green. Agreement shows as a red and a green box on the
    same car; disagreement is immediately obvious. DOTA's oriented box is drawn as a
    rotated quad, so vehicle heading is visible too.

Usage:
    python spikes/compare.py --gallery
    python spikes/compare.py --overlay 1049,4979 --left A --right D
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from yolo_spike import (  # noqa: E402  (path shim must precede this import)
    CONF,
    CONFIGS,
    OUT,
    SRT,
    STRIDE,
    TILE,
    VIDEO,
    gsd_for_altitude,
    nms,
    parse_rel_alt,
    pick_device,
    tile_origins,
)

RED = (0, 0, 255)      # BGR — left/first model
GREEN = (0, 255, 0)    # BGR — right/second model


def detect_with_polys(
    model, frame: np.ndarray, cfg
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Detect and also return oriented-box corners when the model provides them.

    Returns ``(xyxy, conf, polys)`` where ``polys`` has shape (n, 4, 2) for oriented
    models and is ``None`` otherwise. Non-maximum suppression runs on the
    axis-aligned envelopes so both families are merged identically.
    """
    height, width = frame.shape[:2]
    tiles = tile_origins(width, height, TILE, STRIDE) if cfg.tiled else [(0, 0)]
    crops = (
        [frame[y : y + TILE, x : x + TILE] for x, y in tiles] if cfg.tiled else [frame]
    )
    imgsz = TILE if cfg.tiled else cfg.imgsz
    results = model(crops, imgsz=imgsz, conf=CONF, classes=cfg.classes, verbose=False)

    all_xyxy, all_conf, all_poly = [], [], []
    for (ox, oy), res in zip(tiles, results, strict=True):
        container = res.obb if cfg.obb else res.boxes
        if container is None or not len(container):
            continue
        offset = np.array([ox, oy, ox, oy])
        all_xyxy.append(container.xyxy.cpu().numpy() + offset)
        all_conf.append(container.conf.cpu().numpy())
        if cfg.obb:
            all_poly.append(container.xyxyxyxy.cpu().numpy() + np.array([ox, oy]))

    if not all_xyxy:
        return np.empty((0, 4)), np.empty(0), (np.empty((0, 4, 2)) if cfg.obb else None)

    xyxy = np.vstack(all_xyxy)
    conf = np.concatenate(all_conf)
    polys = np.concatenate(all_poly) if cfg.obb else None
    keep = nms(xyxy, conf)
    return xyxy[keep], conf[keep], (polys[keep] if polys is not None else None)


def draw(
    canvas: np.ndarray,
    xyxy: np.ndarray,
    conf: np.ndarray,
    polys: np.ndarray | None,
    colour: tuple[int, int, int],
    label_side: str,
) -> None:
    """
    Draw one model's detections in ``colour``, in place.
    """
    for i, ((x1, y1, x2, y2), c) in enumerate(zip(xyxy, conf, strict=True)):
        if polys is not None:
            cv2.polylines(canvas, [polys[i].astype(np.int32)], True, colour, 2)
        else:
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
        # Offset the two models' labels so overlapping detections stay readable.
        ty = int(y1) - 5 if label_side == "top" else int(y2) + 14
        cv2.putText(
            canvas, f"{c:.2f}", (int(x1), max(12, ty)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA,
        )


def build_overlay(frames: list[int], left_key: str, right_key: str) -> None:
    """
    Render both models onto one image per requested frame.
    """
    from ultralytics import YOLO

    device = pick_device()
    left_cfg, right_cfg = CONFIGS[left_key], CONFIGS[right_key]
    models = {left_key: YOLO(left_cfg.weights), right_key: YOLO(right_cfg.weights)}
    alts = parse_rel_alt(SRT)
    dest = OUT / "overlay"
    dest.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"device {device}   overlay: {left_key}=RED vs {right_key}=GREEN\n")

    for idx in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)  # SRT is 1-based, OpenCV 0-based
        ok, frame = cap.read()
        if not ok:
            print(f"frame {idx}: read failed")
            continue

        lx, lc, lp = detect_with_polys(models[left_key], frame, left_cfg)
        rx, rc, rp = detect_with_polys(models[right_key], frame, right_cfg)

        canvas = frame.copy()
        draw(canvas, lx, lc, lp, RED, "top")
        draw(canvas, rx, rc, rp, GREEN, "bottom")

        alt = alts.get(idx, float("nan"))
        gsd = gsd_for_altitude(alt, width)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            f"frame {idx} | alt {alt:.1f}m | car~{4.5 / gsd:.0f}px | "
            f"RED {left_key}={len(lx)} ({left_cfg.note})   "
            f"GREEN {right_key}={len(rx)} ({right_cfg.note})",
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )
        out = dest / f"overlay_{idx:05d}.jpg"
        cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"frame {idx:>5}  {left_key}={len(lx):>3}  {right_key}={len(rx):>3}  -> {out.name}")

    cap.release()
    print(f"\noverlays -> {dest}")


def build_gallery() -> None:
    """
    Write an HTML page pairing existing annotated frames across configs.
    """
    present = [k for k in CONFIGS if (OUT / k).is_dir() and any((OUT / k).glob("*.jpg"))]
    if not present:
        raise SystemExit(f"no annotated frames under {OUT} — run yolo_spike.py first")

    names = sorted({p.name for k in present for p in (OUT / k).glob("frame_*.jpg")})
    overlays = sorted((OUT / "overlay").glob("*.jpg")) if (OUT / "overlay").is_dir() else []

    rows = []
    for name in names:
        cells = "".join(
            f'<figure><figcaption>{k} — {CONFIGS[k].note}</figcaption>'
            f'<a href="{k}/{name}" target="_blank"><img src="{k}/{name}"></a></figure>'
            for k in present
            if (OUT / k / name).exists()
        )
        rows.append(f"<section><h2>{name}</h2><div class=grid>{cells}</div></section>")

    if overlays:
        cells = "".join(
            f'<figure><figcaption>{p.name} — RED vs GREEN</figcaption>'
            f'<a href="overlay/{p.name}" target="_blank"><img src="overlay/{p.name}"></a>'
            "</figure>"
            for p in overlays
        )
        rows.insert(
            0, f"<section><h2>Two-colour overlays</h2><div class=grid>{cells}</div></section>"
        )

    html = f"""<!doctype html>
<meta charset="utf-8"><title>Detector spike — manual comparison</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px;background:#111;color:#eee}}
 h1{{font-size:20px}} h2{{font-size:15px;color:#9cf;margin:28px 0 8px;font-family:monospace}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:14px}}
 figure{{margin:0}} img{{width:100%;display:block;border:1px solid #333;background:#000}}
 figcaption{{font-family:monospace;font-size:12px;color:#aaa;padding:4px 0}}
 p{{color:#bbb;max-width:70ch}}
</style>
<h1>Detector spike — manual comparison</h1>
<p>Click any image to open it full-size in a new tab (zoom to inspect individual
cars). Captions are burned into each frame: config, altitude, expected car size in
pixels, and detection count. Configs present: {", ".join(present)}.</p>
{"".join(rows)}
"""
    index = OUT / "index.html"
    index.write_text(html, encoding="utf-8")
    print(f"gallery -> {index}\n  open with:  xdg-open {index}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gallery", action="store_true", help="build the HTML comparison page")
    ap.add_argument("--overlay", default="", help="comma-separated frame numbers")
    ap.add_argument("--left", default="A", help="config drawn in RED")
    ap.add_argument("--right", default="D", help="config drawn in GREEN")
    args = ap.parse_args()

    if args.overlay:
        build_overlay(
            [int(v) for v in args.overlay.split(",") if v.strip()], args.left, args.right
        )
    if args.gallery or not args.overlay:
        build_gallery()


if __name__ == "__main__":
    main()
