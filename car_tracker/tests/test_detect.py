"""
Tests for vehicle detection.

The model itself is stubbed throughout: what needs verifying is the tiling, the
coordinate arithmetic that moves tile-local boxes into frame coordinates, and the
de-duplication across tile overlaps. None of that involves a neural network, so these
tests need no weights and no GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from car_tracker.detect import (
    COLUMNS,
    DOTA_VEHICLE_CLASSES,
    DetectionError,
    FrameDetections,
    VehicleDetector,
    frame_range,
    nms,
    resolve_weights,
    tile_origins,
)

WIDTH, HEIGHT = 1920, 1080


class FakeOBB:
    """
    Stands in for ultralytics' oriented-box container.
    """

    def __init__(self, xywhr, conf, cls):
        self._xywhr = np.asarray(xywhr, dtype=float).reshape(-1, 5)
        self._conf = np.asarray(conf, dtype=float)
        self._cls = np.asarray(cls, dtype=int)

    def __len__(self):
        return len(self._xywhr)

    def _wrap(self, array):
        class _Tensor:
            def __init__(self, value):
                self._value = value

            def cpu(self):
                return self

            def numpy(self):
                return self._value

        return _Tensor(array)

    @property
    def xywhr(self):
        return self._wrap(self._xywhr)

    @property
    def conf(self):
        return self._wrap(self._conf)

    @property
    def cls(self):
        return self._wrap(self._cls)

    @property
    def xyxy(self):
        cx, cy, w, h = self._xywhr[:, 0], self._xywhr[:, 1], self._xywhr[:, 2], self._xywhr[:, 3]
        return self._wrap(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1))


class FakeResult:
    def __init__(self, obb=None):
        self.obb = obb


class FakeModel:
    """
    Returns a scripted result per tile and records how it was called.
    """

    def __init__(self, per_tile=None):
        self.per_tile = per_tile or {}
        self.calls: list[dict] = []

    def __call__(self, crops, **kwargs):
        self.calls.append({"n_crops": len(crops), "shapes": [c.shape for c in crops], **kwargs})
        return [FakeResult(self.per_tile.get(i)) for i in range(len(crops))]


@pytest.fixture
def image():
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


class TestTileOrigins:
    def test_covers_frame_with_expected_count(self):
        assert len(tile_origins(WIDTH, HEIGHT)) == 8

    def test_last_tile_is_clamped_not_padded(self):
        """
        Every tile must be full size, so the model never sees black borders.
        """
        origins = tile_origins(WIDTH, HEIGHT)
        assert max(x for x, _ in origins) == WIDTH - 640
        assert max(y for _, y in origins) == HEIGHT - 640

    def test_every_pixel_is_covered(self):
        covered = np.zeros((HEIGHT, WIDTH), dtype=bool)
        for x, y in tile_origins(WIDTH, HEIGHT):
            covered[y : y + 640, x : x + 640] = True
        assert covered.all()

    def test_frame_smaller_than_tile_gives_single_origin(self):
        assert tile_origins(320, 240) == [(0, 0)]

    def test_stride_controls_overlap(self):
        assert len(tile_origins(WIDTH, HEIGHT, stride=320)) > len(tile_origins(WIDTH, HEIGHT))

    @pytest.mark.parametrize(("tile", "stride"), [(0, 512), (640, 0), (-1, -1)])
    def test_rejects_non_positive(self, tile, stride):
        with pytest.raises(ValueError, match="positive"):
            tile_origins(WIDTH, HEIGHT, tile, stride)


class TestNms:
    def test_empty_input(self):
        assert nms(np.empty((0, 4)), np.empty(0)) == []

    def test_keeps_single_box(self):
        assert nms(np.array([[0.0, 0, 10, 10]]), np.array([0.9])) == [0]

    def test_suppresses_duplicate(self):
        boxes = np.array([[0.0, 0, 10, 10], [1, 1, 11, 11]])
        assert nms(boxes, np.array([0.9, 0.5])) == [0]

    def test_keeps_distant_boxes(self):
        boxes = np.array([[0.0, 0, 10, 10], [100, 100, 110, 110]])
        assert sorted(nms(boxes, np.array([0.9, 0.5]))) == [0, 1]

    def test_returns_highest_score_first(self):
        boxes = np.array([[0.0, 0, 10, 10], [100, 100, 110, 110]])
        assert nms(boxes, np.array([0.2, 0.8]))[0] == 1

    def test_zero_area_boxes_do_not_divide_by_zero(self):
        boxes = np.array([[5.0, 5, 5, 5], [5, 5, 5, 5]])
        assert len(nms(boxes, np.array([0.9, 0.8]))) == 2


class TestFrameDetections:
    def test_empty_by_default(self):
        assert len(FrameDetections(frame=1)) == 0
        assert list(FrameDetections(frame=1).rows()) == []

    def test_rows_match_columns(self):
        detections = FrameDetections(
            frame=7,
            centres=np.array([[100.0, 200.0]]),
            sizes=np.array([[70.0, 30.0]]),
            angles_deg=np.array([12.5]),
            confidences=np.array([0.8]),
            classes=np.array([10]),
        )
        row = next(iter(detections.rows()))
        assert tuple(row) == COLUMNS
        assert row["frame"] == 7
        assert row["u"] == 100.0
        assert row["angle_deg"] == 12.5
        assert row["cls"] == 10


class TestDetectFrame:
    def test_batches_all_tiles_in_one_call(self, image):
        model = FakeModel()
        VehicleDetector(model=model).detect_frame(image, frame=1)
        assert len(model.calls) == 1
        assert model.calls[0]["n_crops"] == 8

    def test_passes_confidence_and_classes(self, image):
        model = FakeModel()
        VehicleDetector(model=model, confidence=0.4).detect_frame(image, frame=1)
        assert model.calls[0]["conf"] == 0.4
        assert model.calls[0]["classes"] == list(DOTA_VEHICLE_CLASSES)

    def test_tiles_are_all_full_size(self, image):
        model = FakeModel()
        VehicleDetector(model=model).detect_frame(image, frame=1)
        assert all(shape[:2] == (640, 640) for shape in model.calls[0]["shapes"])

    def test_no_detections_returns_empty(self, image):
        result = VehicleDetector(model=FakeModel()).detect_frame(image, frame=3)
        assert len(result) == 0
        assert result.frame == 3

    def test_tile_local_coords_are_shifted_to_frame(self, image):
        """
        Tile 1 starts at x=512, so a box at tile x=100 sits at frame x=612.
        """
        model = FakeModel({1: FakeOBB([[100, 50, 70, 30, 0.0]], [0.9], [10])})
        result = VehicleDetector(model=model).detect_frame(image, frame=1)
        assert result.centres[0, 0] == pytest.approx(512 + 100)
        assert result.centres[0, 1] == pytest.approx(50)

    def test_angle_converted_to_degrees(self, image):
        model = FakeModel({0: FakeOBB([[10, 10, 70, 30, np.pi / 2]], [0.9], [10])})
        result = VehicleDetector(model=model).detect_frame(image, frame=1)
        assert result.angles_deg[0] == pytest.approx(90.0)

    def test_sizes_and_classes_preserved(self, image):
        model = FakeModel({0: FakeOBB([[10, 10, 70, 30, 0.0]], [0.7], [9])})
        result = VehicleDetector(model=model).detect_frame(image, frame=1)
        assert tuple(result.sizes[0]) == pytest.approx((70.0, 30.0))
        assert result.classes[0] == 9
        assert result.confidences[0] == pytest.approx(0.7)

    def test_duplicate_across_overlapping_tiles_is_merged(self, image):
        """
        Tiles 0 and 1 overlap on x in [512, 640); a car there is seen twice and must
        survive only once.
        """
        model = FakeModel(
            {
                0: FakeOBB([[600, 100, 70, 30, 0.0]], [0.9], [10]),
                1: FakeOBB([[88, 100, 70, 30, 0.0]], [0.6], [10]),  # same frame position
            }
        )
        result = VehicleDetector(model=model).detect_frame(image, frame=1)
        assert len(result) == 1
        assert result.confidences[0] == pytest.approx(0.9)

    def test_distinct_cars_in_different_tiles_both_kept(self, image):
        model = FakeModel(
            {
                0: FakeOBB([[100, 100, 70, 30, 0.0]], [0.9], [10]),
                5: FakeOBB([[300, 300, 70, 30, 0.0]], [0.8], [10]),
            }
        )
        assert len(VehicleDetector(model=model).detect_frame(image, frame=1)) == 2

    def test_multiple_boxes_in_one_tile(self, image):
        model = FakeModel(
            {0: FakeOBB([[50, 50, 70, 30, 0.0], [400, 400, 70, 30, 0.0]], [0.9, 0.8], [10, 9])}
        )
        assert len(VehicleDetector(model=model).detect_frame(image, frame=1)) == 2

    def test_tile_without_obb_attribute_is_skipped(self, image):
        model = FakeModel({0: None})
        assert len(VehicleDetector(model=model).detect_frame(image, frame=1)) == 0


class TestDetectVideo:
    @pytest.fixture
    def clip(self, tmp_path):
        import cv2

        path = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (WIDTH, HEIGHT))
        for i in range(5):
            writer.write(np.full((HEIGHT, WIDTH, 3), 10 + i, dtype=np.uint8))
        writer.release()
        return path

    def test_returns_dataframe_with_columns(self, clip):
        model = FakeModel({0: FakeOBB([[50, 50, 70, 30, 0.0]], [0.9], [10])})
        frame = VehicleDetector(model=model).detect_video(clip, [1, 2], progress_every=0)
        assert list(frame.columns) == list(COLUMNS)
        assert len(frame) == 2

    def test_frame_numbers_recorded(self, clip):
        model = FakeModel({0: FakeOBB([[50, 50, 70, 30, 0.0]], [0.9], [10])})
        frame = VehicleDetector(model=model).detect_video(clip, [3, 1], progress_every=0)
        assert sorted(frame["frame"].tolist()) == [1, 3]

    def test_writes_csv_incrementally(self, clip, tmp_path):
        """
        A long run that dies near the end must leave earlier frames on disk.
        """
        out = tmp_path / "sub" / "detections.csv"
        model = FakeModel({0: FakeOBB([[50, 50, 70, 30, 0.0]], [0.9], [10])})
        VehicleDetector(model=model).detect_video(clip, [1, 2], out_path=out, progress_every=0)
        written = pd.read_csv(out)
        assert list(written.columns) == list(COLUMNS)
        assert len(written) == 2

    def test_csv_has_header_even_with_no_detections(self, clip, tmp_path):
        out = tmp_path / "empty.csv"
        VehicleDetector(model=FakeModel()).detect_video(
            clip, [1], out_path=out, progress_every=0
        )
        assert pd.read_csv(out).empty
        assert list(pd.read_csv(out).columns) == list(COLUMNS)

    def test_no_frames_raises(self, clip):
        with pytest.raises(DetectionError, match="no frames"):
            VehicleDetector(model=FakeModel()).detect_video(clip, [], progress_every=0)

    def test_output_feeds_geo_projection(self, clip):
        """
        The contract that matters: detections must drop straight into geo.py.
        """
        from car_tracker.geo import CameraModel

        model = FakeModel({0: FakeOBB([[50, 50, 70, 30, 0.0]], [0.9], [10])})
        detections = VehicleDetector(model=model).detect_video(clip, [1], progress_every=0)
        telemetry = pd.DataFrame(
            {
                "frame": [1],
                "t_sec": [0.0],
                "lat": [48.267],
                "lon": [25.9145],
                "rel_alt": [102.0],
                "gsd": [0.08],
                "yaw": [-65.8],
            }
        )
        projected = CameraModel().project_detections(detections, telemetry)
        assert {"lat", "lon"} <= set(projected.columns)
        assert projected["lat"].notna().all()


class TestFrameRange:
    def test_all_frames_by_default(self):
        assert frame_range(5) == [1, 2, 3, 4, 5]

    def test_stride(self):
        assert frame_range(10, stride=3) == [1, 4, 7, 10]

    def test_limit(self):
        assert frame_range(100, stride=1, limit=3) == [1, 2, 3]

    def test_is_one_based(self):
        assert frame_range(3)[0] == 1

    @pytest.mark.parametrize("stride", [0, -1])
    def test_rejects_bad_stride(self, stride):
        with pytest.raises(ValueError, match="stride"):
            frame_range(10, stride=stride)


class TestResolveWeights:
    """
    Weights must land in one predictable place rather than the current directory,
    which is where ultralytics would otherwise download 110 MB files.
    """

    def test_bare_name_goes_to_the_models_directory(self, tmp_path):
        assert resolve_weights("yolo11x-obb.pt", tmp_path) == str(tmp_path / "yolo11x-obb.pt")

    def test_directory_is_created_so_the_download_has_somewhere_to_land(self, tmp_path):
        target = tmp_path / "models"
        resolve_weights("yolo11n.pt", target)
        assert target.is_dir()

    def test_explicit_path_is_passed_through(self, tmp_path):
        assert resolve_weights("/opt/models/custom.pt", tmp_path) == "/opt/models/custom.pt"

    def test_existing_file_in_the_working_directory_wins(self, tmp_path, monkeypatch):
        local = tmp_path / "local.pt"
        local.write_bytes(b"")
        monkeypatch.chdir(tmp_path)
        assert resolve_weights("local.pt", tmp_path / "models") == "local.pt"
