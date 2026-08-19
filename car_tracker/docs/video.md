
# video

`car_tracker/src/car_tracker/video.py` — frame access. Implemented.

A thin wrapper over OpenCV with two jobs: guarantee `release()` is called, and state
the frame-numbering convention in exactly one place.

## API

```python
from car_tracker.video import probe, read_frame, read_frames, video_capture

info = probe("video2.MP4")            # 1920x1080 @ 29.97, 4979 frames
with video_capture("video2.MP4") as capture:
    frame = read_frame(capture, 1049)  # 1-based
for number, image in read_frames("video2.MP4", [100, 200, 300]):
    ...
```

| Member | Does |
|---|---|
| `probe(path)` | `VideoInfo(width, height, fps, frame_count)` without decoding |
| `video_capture(path)` | context manager; releases even on exception |
| `read_frame(capture, n)` | one frame by **1-based** number |
| `read_frames(path, numbers)` | iterate frames, ascending |
| `VideoError` | cannot open, or cannot decode a frame |

## Frame numbering

The SRT counts frames from **1** (`FrameCnt: 1`); OpenCV indexes from **0**.
`read_frame` takes the SRT convention and does the conversion, so the off-by-one
exists in one function rather than at every call site.

Frame numbers below 1 raise rather than silently reading the wrong frame.

## Notes

- `read_frames` visits numbers in ascending order and deduplicates, so the decoder
  seeks forward only — markedly faster than jumping around a 656 MB file.
- OpenCV bundles its own decoders, so no system `ffmpeg` is required.

## Tests

`car_tracker/tests/test_video.py` — 14 tests against a generated clip whose Nth frame
has intensity N×20, so frame identity is verifiable from pixel values. Covers 1-based
numbering, release on exception, ascending traversal, and error cases. Tolerances are
loose because mp4v is lossy; the assertion is which frame came back, not its exact
intensity.
