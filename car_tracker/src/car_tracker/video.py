"""
Video frame access.

Thin wrapper over OpenCV so the rest of the package never has to remember to call
``release()``, and so frame numbering is stated once: the SRT counts frames from 1,
OpenCV indexes from 0.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class VideoError(RuntimeError):
    """
    Raised when a video cannot be opened or a requested frame cannot be read.
    """


@dataclass(frozen=True)
class VideoInfo:
    """
    Basic properties of an opened video.
    """

    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


@contextmanager
def video_capture(path: str | Path) -> Iterator[cv2.VideoCapture]:
    """
    Open a video, guaranteeing release even if the caller raises.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoError(f"cannot open video: {path}")
    try:
        yield capture
    finally:
        capture.release()


def probe(path: str | Path) -> VideoInfo:
    """
    Read width, height, fps and frame count without decoding any frames.
    """
    with video_capture(path) as capture:
        return VideoInfo(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )


def read_frame(capture: cv2.VideoCapture, frame_number: int) -> np.ndarray:
    """
    Read one frame by 1-based frame number, as used by the SRT's ``FrameCnt``.

    Raises:
        VideoError: if the frame cannot be decoded.
    """
    if frame_number < 1:
        raise VideoError(f"frame numbers are 1-based, got {frame_number}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise VideoError(f"cannot read frame {frame_number}")
    return frame


def read_frames(
    path: str | Path, frame_numbers: Sequence[int]
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield ``(frame_number, image)`` for the requested 1-based frame numbers.

    Numbers are visited in ascending order so the decoder seeks forward only, which
    is markedly faster on long files than jumping around.
    """
    with video_capture(path) as capture:
        for number in sorted(set(frame_numbers)):
            yield number, read_frame(capture, number)
