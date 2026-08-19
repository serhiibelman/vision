"""
Detect vehicles in nadir drone frames.

Uses YOLO11 with DOTA-trained oriented-box weights. DOTA is aerial imagery, so unlike
COCO it has actually seen vehicles from directly overhead; COCO weights scored snowy
roofs as cars and reached only ~29% recall on this footage (DECISIONS.md D1).

Frames are processed as overlapping 640x640 tiles at native resolution rather than
resized whole. At the library default ``imgsz=640`` a 1920-wide frame is downscaled 3x
and a car shrinks from ~57 px to ~19 px, which collapsed detections from 61 to 4
across a 20-frame sample.

Output is one row per detection with the oriented box centroid in pixels, ready for
:meth:`car_tracker.geo.CameraModel.project_detections`.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from car_tracker.video import read_frame, video_capture

# DOTA class ids: 9 = large vehicle, 10 = small vehicle. Both are kept; "car" versus
# "truck" is not a distinction this footage supports at ~57 px.
DOTA_VEHICLE_CLASSES = (9, 10)

DEFAULT_WEIGHTS = "yolo11x-obb.pt"

# Ultralytics resolves a bare weights name against the current working directory and
# downloads there on a miss, so the 100+ MB files would land wherever the command was
# run from. Pin one location instead; override with CAR_TRACKER_MODELS.
DEFAULT_WEIGHTS_DIR = Path(
    os.environ.get("CAR_TRACKER_MODELS")
    or Path.home() / ".cache" / "car-tracker" / "models"
)
DEFAULT_CONFIDENCE = 0.25
TILE_SIZE = 640
TILE_STRIDE = 512  # ~20% overlap, so a car near a seam is whole in one tile
NMS_IOU = 0.5

COLUMNS = ("frame", "u", "v", "conf", "cls", "w_px", "h_px", "angle_deg")


class DetectionError(RuntimeError):
    """
    Raised when detection cannot proceed.
    """


def tile_origins(
    width: int, height: int, tile: int = TILE_SIZE, stride: int = TILE_STRIDE
) -> list[tuple[int, int]]:
    """
    Top-left corners of tiles covering the frame.

    The last row and column are clamped inward rather than padded, so every tile is
    full size and the model never sees black borders. The cost is extra overlap at the
    edges, which NMS removes anyway.
    """
    if tile <= 0 or stride <= 0:
        raise ValueError("tile and stride must be positive")

    def axis(total: int) -> list[int]:
        if total <= tile:
            return [0]
        origins = list(range(0, total - tile + 1, stride))
        if origins[-1] != total - tile:
            origins.append(total - tile)
        return origins

    return [(x, y) for y in axis(height) for x in axis(width)]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = NMS_IOU) -> list[int]:
    """
    Greedy non-maximum suppression over axis-aligned xyxy boxes.

    Needed because overlapping tiles see the same car twice. Hand-rolled to keep the
    package free of a torchvision dependency.

    Returns:
        Indices to keep, highest score first.
    """
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = np.asarray(boxes, dtype=float).T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.asarray(scores, dtype=float).argsort()[::-1]

    keep: list[int] = []
    while order.size:
        current = order[0]
        keep.append(int(current))
        if order.size == 1:
            break
        rest = order[1:]
        left = np.maximum(x1[current], x1[rest])
        top = np.maximum(y1[current], y1[rest])
        right = np.minimum(x2[current], x2[rest])
        bottom = np.minimum(y2[current], y2[rest])
        overlap = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
        union = areas[current] + areas[rest] - overlap
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, overlap / union, 0.0)
        order = rest[iou <= iou_threshold]
    return keep


@dataclass
class FrameDetections:
    """
    Oriented-box detections for one frame, in full-frame pixel coordinates.

    Attributes:
        frame: 1-based frame number.
        centres: (n, 2) array of box centroids.
        sizes: (n, 2) array of box width and height in pixels. Diagnostic only —
            these undersize real vehicles by ~25% (DECISIONS.md D5).
        angles_deg: (n,) box rotation in degrees.
        confidences: (n,) detection scores.
        classes: (n,) DOTA class ids.
    """

    frame: int
    centres: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    sizes: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    angles_deg: np.ndarray = field(default_factory=lambda: np.empty(0))
    confidences: np.ndarray = field(default_factory=lambda: np.empty(0))
    classes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    def __len__(self) -> int:
        return len(self.centres)

    def rows(self) -> Iterator[dict]:
        """
        Yield one flat record per detection, matching :data:`COLUMNS`.
        """
        for i in range(len(self)):
            yield {
                "frame": self.frame,
                "u": float(self.centres[i, 0]),
                "v": float(self.centres[i, 1]),
                "conf": float(self.confidences[i]),
                "cls": int(self.classes[i]),
                "w_px": float(self.sizes[i, 0]),
                "h_px": float(self.sizes[i, 1]),
                "angle_deg": float(self.angles_deg[i]),
            }


def resolve_weights(weights: str | Path, directory: Path = DEFAULT_WEIGHTS_DIR) -> str:
    """
    Turn a weights name into an explicit path under ``directory``.

    An existing path or anything containing a separator is passed through untouched, so
    a caller can still point at weights anywhere. A bare name resolves into the models
    directory, which is created so ultralytics downloads into it rather than the cwd.
    """
    candidate = Path(weights)
    if candidate.exists() or len(candidate.parts) > 1:
        return str(candidate)

    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / candidate)


def pick_device() -> int | str:
    """
    CUDA if present, else Apple MPS, else CPU.

    Never hardcoded: development happens on a CPU laptop while full runs happen on an
    NVIDIA machine, and a reviewer may have neither.
    """
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise DetectionError(
            "torch is not installed; install the 'detect' extra: pip install -e '.[detect]'"
        ) from error

    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class VehicleDetector:
    """
    Tiled oriented-box vehicle detector.

    Args:
        weights: ultralytics weights name or path. Defaults to DOTA-trained OBB.
            A bare name resolves under :data:`DEFAULT_WEIGHTS_DIR`.
        confidence: score threshold. Deliberately permissive — temporal consistency
            across frames filters false positives better than a high threshold, and
            detections never made cannot be recovered.
        classes: DOTA class ids to keep.
        tile: tile edge length in pixels.
        stride: distance between tile origins.
        device: overrides auto-detection.
        model: pre-built model, primarily for testing without weights on disk.
    """

    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS,
        confidence: float = DEFAULT_CONFIDENCE,
        classes: Sequence[int] = DOTA_VEHICLE_CLASSES,
        tile: int = TILE_SIZE,
        stride: int = TILE_STRIDE,
        device: int | str | None = None,
        model: object | None = None,
    ) -> None:
        self.confidence = confidence
        self.classes = list(classes)
        self.tile = tile
        self.stride = stride
        self.device = device
        self._model = model
        self._weights = weights

    @property
    def model(self) -> Any:
        """
        The underlying model, loaded on first use so construction stays cheap.
        """
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise DetectionError(
                    "ultralytics is not installed; install the 'detect' extra: "
                    "pip install -e '.[detect]'"
                ) from error
            self._model = YOLO(resolve_weights(self._weights))
            if self.device is None:
                self.device = pick_device()
        return self._model

    def detect_frame(self, image: np.ndarray, frame: int) -> FrameDetections:
        """
        Detect vehicles in one full-resolution frame.

        Tiles are batched into a single model call, then results are shifted back into
        full-frame coordinates and de-duplicated across tile overlaps.
        """
        height, width = image.shape[:2]
        origins = tile_origins(width, height, self.tile, self.stride)
        crops = [image[y : y + self.tile, x : x + self.tile] for x, y in origins]

        results = self.model(
            crops,
            imgsz=self.tile,
            conf=self.confidence,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        boxes, centres, sizes, angles, confidences, classes = [], [], [], [], [], []
        for (origin_x, origin_y), result in zip(origins, results, strict=True):
            oriented = getattr(result, "obb", None)
            if oriented is None or not len(oriented):
                continue
            offset = np.array([origin_x, origin_y], dtype=float)
            xywhr = np.asarray(oriented.xywhr.cpu().numpy(), dtype=float)

            centres.append(xywhr[:, :2] + offset)
            sizes.append(xywhr[:, 2:4])
            angles.append(np.degrees(xywhr[:, 4]))
            confidences.append(np.asarray(oriented.conf.cpu().numpy(), dtype=float))
            classes.append(np.asarray(oriented.cls.cpu().numpy(), dtype=int))
            # Axis-aligned envelope, used only to decide which duplicates to drop.
            boxes.append(np.asarray(oriented.xyxy.cpu().numpy(), dtype=float) + np.tile(offset, 2))

        if not centres:
            return FrameDetections(frame=frame)

        keep = nms(np.vstack(boxes), np.concatenate(confidences))
        return FrameDetections(
            frame=frame,
            centres=np.vstack(centres)[keep],
            sizes=np.vstack(sizes)[keep],
            angles_deg=np.concatenate(angles)[keep],
            confidences=np.concatenate(confidences)[keep],
            classes=np.concatenate(classes)[keep],
        )

    def detect_video(
        self,
        video_path: str | Path,
        frames: Iterable[int],
        out_path: str | Path | None = None,
        progress_every: int = 50,
    ) -> pd.DataFrame:
        """
        Detect across many frames, optionally streaming results to CSV.

        Rows are flushed as each frame completes, so a long run that crashes near the
        end still leaves everything detected so far on disk.

        Returns:
            All detections as a DataFrame with :data:`COLUMNS`.
        """
        wanted = sorted(set(int(f) for f in frames))
        if not wanted:
            raise DetectionError("no frames requested")

        rows: list[dict] = []
        handle = None
        writer = None
        if out_path is not None:
            destination = Path(out_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            handle = destination.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
            writer.writeheader()

        try:
            with video_capture(video_path) as capture:
                for position, number in enumerate(wanted, start=1):
                    detections = self.detect_frame(read_frame(capture, number), number)
                    frame_rows = list(detections.rows())
                    rows.extend(frame_rows)
                    if writer is not None and handle is not None:
                        writer.writerows(frame_rows)
                        handle.flush()
                    if progress_every and position % progress_every == 0:
                        print(
                            f"{position}/{len(wanted)} frames  {len(rows)} detections",
                            flush=True,
                        )
        finally:
            if handle is not None:
                handle.close()

        return pd.DataFrame(rows, columns=list(COLUMNS))


def frame_range(frame_count: int, stride: int = 1, limit: int | None = None) -> list[int]:
    """
    1-based frame numbers to process.

    Args:
        frame_count: total frames in the video.
        stride: keep every Nth frame. A GPU can afford stride 1; a CPU cannot.
        limit: stop after this many frames.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    numbers = list(range(1, frame_count + 1, stride))
    return numbers[:limit] if limit else numbers
