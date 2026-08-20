"""
Video frame access.

Thin wrapper over OpenCV so the rest of the package never has to remember to call
``release()``, and so frame numbering is stated once: the SRT counts frames from 1,
OpenCV indexes from 0.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import cv2
import numpy as np

T = TypeVar("T")


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


# Walking forward with grab() beats seeking until the gap grows past roughly one group
# of pictures; past that, letting the decoder jump to the next keyframe wins. Measured
# on video2: walking is 3-4.5x faster at gap 1, 1.3-1.6x at gap 15-20, and slower from
# gap 30 on.
SEEK_GAP = 20


def read_frames_from(
    capture: cv2.VideoCapture, frame_numbers: Sequence[int]
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield ``(frame_number, image)`` for 1-based frame numbers, decoding each frame once.

    :func:`read_frame` seeks on every call, and a seek on inter-frame-coded video means
    jumping back to the preceding keyframe and re-decoding everything in between. At
    stride 1 that re-decodes most of a GOP per frame. Here the stream is walked forward
    with ``grab()``, which decodes without the colour conversion, and only frames that
    were actually asked for are retrieved. Seeking is kept for gaps wider than
    :data:`SEEK_GAP`, where re-decoding from a keyframe is the cheaper of the two.

    Numbers are visited in ascending order; duplicates are dropped.
    """
    position = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    for number in sorted(set(frame_numbers)):
        if number < 1:
            raise VideoError(f"frame numbers are 1-based, got {number}")
        target = number - 1
        if target < position or target - position > SEEK_GAP:
            capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            position = target
        while position < target:
            if not capture.grab():
                raise VideoError(f"cannot read frame {position + 1}")
            position += 1
        ok, frame = capture.read()
        if not ok or frame is None:
            raise VideoError(f"cannot read frame {number}")
        position += 1
        yield number, frame


def read_frames(
    path: str | Path, frame_numbers: Sequence[int]
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Open ``path`` and yield ``(frame_number, image)`` for the requested frame numbers.
    """
    with video_capture(path) as capture:
        yield from read_frames_from(capture, frame_numbers)


# Frames decoded ahead of the consumer. Deep enough to cover a decode that runs long,
# shallow enough that a 1920x1080 backlog stays under ~25 MB.
PREFETCH_DEPTH = 4


def prefetch(items: Iterable[T], depth: int = PREFETCH_DEPTH) -> Iterator[T]:
    """
    Yield from ``items``, producing them on a worker thread a few steps ahead.

    Decoding and inference are otherwise strictly alternating: the GPU idles through
    every ``grab()`` and the decoder idles through every forward pass. Both OpenCV and
    torch release the GIL while working, so a plain thread genuinely overlaps them.

    The queue is bounded, so a fast producer cannot run away with memory. Exceptions
    raised by the producer surface here, at the point of consumption. Abandoning the
    iterator early stops the worker rather than leaking it.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")

    # Each entry is (payload, error); payload is None only in the end marker, since
    # real payloads come from the wrapped iterator.
    buffer: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def put(entry: tuple) -> bool:
        """
        Hand one entry over, reporting whether the consumer is still listening.

        The put blocks rather than polls, so the producer wakes the instant the
        consumer frees a slot. A consumer that abandons the iterator drains the queue,
        which is what releases a producer parked here.
        """
        if stop.is_set():
            return False
        buffer.put(entry)
        return not stop.is_set()

    def produce() -> None:
        try:
            for item in items:
                if not put((item, None)):
                    return
        except Exception as error:  # re-raised on the consuming thread
            put((None, error))
        else:
            put((None, None))

    worker = threading.Thread(target=produce, name="prefetch", daemon=True)
    worker.start()
    try:
        while True:
            payload, error = buffer.get()
            if error is not None:
                raise error
            if payload is None:
                return
            yield payload
    finally:
        stop.set()
        # Draining is what wakes a producer blocked on a full queue.
        while worker.is_alive():
            try:
                buffer.get_nowait()
            except queue.Empty:
                worker.join(timeout=0.05)
