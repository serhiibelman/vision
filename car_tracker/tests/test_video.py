"""
Tests for video frame access.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest

from car_tracker.video import (
    VideoError,
    prefetch,
    probe,
    read_frame,
    read_frames,
    read_frames_from,
    video_capture,
)

WIDTH, HEIGHT, FPS, COUNT = 64, 48, 30.0, 10


@pytest.fixture
def clip(tmp_path):
    """
    A short clip whose Nth frame is filled with intensity N*20, so frame identity is
    checkable from pixel values.
    """
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    for i in range(1, COUNT + 1):
        writer.write(np.full((HEIGHT, WIDTH, 3), i * 20, dtype=np.uint8))
    writer.release()
    return path


class TestProbe:
    def test_reads_properties(self, clip):
        info = probe(clip)
        assert (info.width, info.height) == (WIDTH, HEIGHT)
        assert info.fps == pytest.approx(FPS)
        assert info.frame_count == COUNT

    def test_size_and_duration(self, clip):
        info = probe(clip)
        assert info.size == (WIDTH, HEIGHT)
        assert info.duration_s == pytest.approx(COUNT / FPS)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(VideoError, match="cannot open"):
            probe(tmp_path / "nope.mp4")


class TestVideoCapture:
    def test_releases_on_exception(self, clip):
        """
        The whole reason this is a context manager.
        """
        with pytest.raises(RuntimeError, match="boom"):
            with video_capture(clip) as capture:
                assert capture.isOpened()
                raise RuntimeError("boom")

    def test_released_after_block(self, clip):
        with video_capture(clip) as capture:
            pass
        assert not capture.isOpened()


class TestReadFrame:
    def test_numbering_is_one_based(self, clip):
        """
        Frame 1 is the first frame, matching the SRT's FrameCnt.

        Tolerances are loose because mp4v is lossy; the point is which frame came
        back, not its exact intensity.
        """
        with video_capture(clip) as capture:
            assert read_frame(capture, 1).mean() == pytest.approx(20, abs=8)

    def test_reads_a_later_frame(self, clip):
        with video_capture(clip) as capture:
            assert read_frame(capture, 5).mean() == pytest.approx(100, abs=8)

    def test_frames_are_distinguishable(self, clip):
        """
        Seeking must actually land on different frames, not repeat one.
        """
        with video_capture(clip) as capture:
            means = [read_frame(capture, n).mean() for n in (1, 5, 9)]
        assert means[0] < means[1] < means[2]

    @pytest.mark.parametrize("number", [0, -1])
    def test_rejects_non_positive(self, clip, number):
        with video_capture(clip) as capture:
            with pytest.raises(VideoError, match="1-based"):
                read_frame(capture, number)

    def test_beyond_end_raises(self, clip):
        with video_capture(clip) as capture:
            with pytest.raises(VideoError, match="cannot read"):
                read_frame(capture, COUNT + 50)


class TestReadFrames:
    def test_yields_requested_frames(self, clip):
        got = dict(read_frames(clip, [3, 1]))
        assert set(got) == {1, 3}

    def test_ascending_order_regardless_of_input(self, clip):
        assert [n for n, _ in read_frames(clip, [7, 2, 5])] == [2, 5, 7]

    def test_deduplicates(self, clip):
        assert [n for n, _ in read_frames(clip, [4, 4, 4])] == [4]


class TestReadFramesFrom:
    def test_identifies_frames_correctly(self, clip):
        """
        The sequential walk must land on the same frames as the seeking reader.
        """
        with video_capture(clip) as capture:
            got = {n: image.mean() for n, image in read_frames_from(capture, [1, 2, 5, 9])}
        for number, mean in got.items():
            assert mean == pytest.approx(number * 20, abs=8)

    def test_matches_read_frame(self, clip):
        with video_capture(clip) as capture:
            sequential = [image.mean() for _, image in read_frames_from(capture, [2, 3, 4])]
        with video_capture(clip) as capture:
            seeking = [read_frame(capture, n).mean() for n in (2, 3, 4)]
        assert sequential == pytest.approx(seeking, abs=1)

    def test_survives_a_backward_request(self, clip):
        """
        Numbers are sorted, but a caller reusing a capture may start mid-stream.
        """
        with video_capture(clip) as capture:
            read_frame(capture, 8)
            got = [n for n, _ in read_frames_from(capture, [2, 6])]
        assert got == [2, 6]

    def test_rejects_zero(self, clip):
        with video_capture(clip) as capture:
            with pytest.raises(VideoError, match="1-based"):
                list(read_frames_from(capture, [0]))

    def test_beyond_end_raises(self, clip):
        with video_capture(clip) as capture:
            with pytest.raises(VideoError, match="cannot read"):
                list(read_frames_from(capture, [COUNT + 50]))


class TestPrefetch:
    def test_preserves_order_and_contents(self):
        assert list(prefetch(iter(range(20)), depth=3)) == list(range(20))

    def test_empty_source(self):
        assert list(prefetch(iter([]))) == []

    def test_propagates_producer_errors(self):
        def failing():
            yield 1
            raise VideoError("decode blew up")

        with pytest.raises(VideoError, match="decode blew up"):
            list(prefetch(failing(), depth=2))

    def test_abandoning_early_stops_the_worker(self):
        """
        Breaking out of the loop must not leave a thread blocked on a full queue.
        """
        before = threading.active_count()
        for value in prefetch(iter(range(1000)), depth=2):
            if value == 3:
                break
        deadline = time.monotonic() + 2.0
        while threading.active_count() > before and time.monotonic() < deadline:
            time.sleep(0.02)
        assert threading.active_count() == before

    def test_rejects_zero_depth(self):
        with pytest.raises(ValueError, match="depth"):
            list(prefetch(iter([1]), depth=0))
