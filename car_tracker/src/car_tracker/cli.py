
"""
Command line entry point.

Each stage reads files and writes files, so the expensive detection pass runs once
while the cheap stages downstream are re-run freely. Detection over 4979 frames takes
minutes on a GPU and hours on a CPU; re-drawing the map takes a second.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from car_tracker.calibrate import calibrate
from car_tracker.detect import VehicleDetector, frame_range
from car_tracker.geo import Calibration, CameraModel
from car_tracker.postprocess import moving_tracks, report, to_geojson
from car_tracker.telemetry import load as load_telemetry
from car_tracker.telemetry import summarise as summarise_telemetry
from car_tracker.track import track_detections
from car_tracker.video import probe
from car_tracker.visualization import render_flight_path, render_tracks

DEFAULT_VIDEO = Path("../tech-assignment/video2.MP4")
DEFAULT_SRT = Path("../tech-assignment/video2.SRT")
OUTPUTS = Path("outputs")
RESULTS = Path("results")


def save_calibration(calibration: Calibration, path: str | Path) -> Path:
    """
    Persist calibration so detection and tracking need not re-measure it.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(calibration), indent=2), encoding="utf-8")
    return destination


def load_calibration(path: str | Path | None) -> Calibration:
    """
    Read calibration from disk, falling back to uncalibrated defaults.
    """
    if path is None or not Path(path).exists():
        return Calibration()
    return Calibration(**json.loads(Path(path).read_text(encoding="utf-8")))


def _frames(args, frame_count: int) -> list[int]:
    """
    Resolve the requested frame numbers from CLI arguments.
    """
    numbers = frame_range(frame_count, stride=args.stride, limit=args.limit)
    start, end = args.start, args.end or frame_count
    return [n for n in numbers if start <= n <= end]


def cmd_telemetry(args) -> None:
    telemetry = load_telemetry(args.srt)
    print(summarise_telemetry(telemetry))
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    telemetry.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    if args.map:
        print(f"wrote {render_flight_path(telemetry, args.map)}")


def cmd_calibrate(args) -> None:
    info = probe(args.video)
    telemetry = load_telemetry(args.srt, expected_frames=info.frame_count)
    result = calibrate(args.video, telemetry, image_size=info.size, count=args.pairs)
    print(result.report())
    print(f"\nwrote {save_calibration(result.calibration, args.out)}")


def cmd_detect(args) -> None:
    info = probe(args.video)
    frames = _frames(args, info.frame_count)
    detector = VehicleDetector(weights=args.weights, confidence=args.conf)
    print(f"{len(frames)} frames, weights {args.weights}")
    detections = detector.detect_video(args.video, frames, out_path=args.out)
    print(f"\n{len(detections)} detections -> {args.out}")


def cmd_track(args) -> None:
    info = probe(args.video)
    telemetry = load_telemetry(args.srt, expected_frames=info.frame_count)
    detections = pd.read_csv(args.detections)
    model = CameraModel(image_size=info.size, calibration=load_calibration(args.calibration))

    projected = model.project_detections(detections, telemetry)
    tracks = track_detections(projected)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(args.out, index=False)
    print(f"{len(detections)} detections -> {tracks['track_id'].nunique()} tracks -> {args.out}")


def cmd_map(args) -> None:
    tracks = pd.read_csv(args.tracks)
    moving, features = moving_tracks(tracks)
    print(report(features))

    RESULTS.mkdir(parents=True, exist_ok=True)
    telemetry = load_telemetry(args.srt) if args.srt else None
    if telemetry is not None and not tracks.empty:
        first, last = tracks["frame"].min(), tracks["frame"].max()
        telemetry = telemetry[telemetry["frame"].between(first, last)]

    print(f"\nwrote {to_geojson(moving, features, args.geojson)}")
    drawn = render_tracks(moving, features, args.out, telemetry=telemetry, all_tracks=tracks)
    print(f"wrote {drawn}")
    features.to_csv(OUTPUTS / "track_features.csv", index=False)


def cmd_all(args) -> None:
    """
    Run every stage in order, reusing an existing calibration when present.
    """
    info = probe(args.video)
    telemetry = load_telemetry(args.srt, expected_frames=info.frame_count)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    telemetry.to_csv(OUTPUTS / "telemetry.csv", index=False)

    calibration_path = OUTPUTS / "calibration.json"
    if calibration_path.exists() and not args.recalibrate:
        calibration = load_calibration(calibration_path)
        print(f"reusing calibration from {calibration_path}")
    else:
        result = calibrate(args.video, telemetry, image_size=info.size, count=args.pairs)
        print(result.report())
        calibration = result.calibration
        save_calibration(calibration, calibration_path)
    print()

    frames = _frames(args, info.frame_count)
    detector = VehicleDetector(weights=args.weights, confidence=args.conf)
    detections = detector.detect_video(
        args.video, frames, out_path=OUTPUTS / "detections.csv"
    )
    print(f"{len(detections)} detections\n")

    model = CameraModel(image_size=info.size, calibration=calibration)
    tracks = track_detections(model.project_detections(detections, telemetry))
    tracks.to_csv(OUTPUTS / "tracks.csv", index=False)

    moving, features = moving_tracks(tracks)
    features.to_csv(OUTPUTS / "track_features.csv", index=False)
    print(report(features))

    RESULTS.mkdir(parents=True, exist_ok=True)
    segment = telemetry[telemetry["frame"].isin(frames)]
    print(f"\nwrote {to_geojson(moving, features, RESULTS / 'tracks.geojson')}")
    drawn = render_tracks(
        moving, features, RESULTS / "map.html", telemetry=segment, all_tracks=tracks
    )
    print(f"wrote {drawn}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="car-tracker", description="Track moving cars in drone footage and map them."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub, video=True, srt=True):
        if video:
            sub.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
        if srt:
            sub.add_argument("--srt", type=Path, default=DEFAULT_SRT)

    def add_frame_args(sub):
        sub.add_argument("--stride", type=int, default=1, help="process every Nth frame")
        sub.add_argument("--start", type=int, default=1, help="first frame")
        sub.add_argument("--end", type=int, default=None, help="last frame")
        sub.add_argument("--limit", type=int, default=None, help="stop after N frames")

    def add_detector_args(sub):
        sub.add_argument("--weights", default="yolo11x-obb.pt")
        sub.add_argument("--conf", type=float, default=0.25)

    telemetry = subparsers.add_parser("telemetry", help="parse the SRT sidecar")
    add_common(telemetry, video=False)
    telemetry.add_argument("--out", type=Path, default=OUTPUTS / "telemetry.csv")
    telemetry.add_argument("--map", type=Path, default=None, help="also draw the flight path")
    telemetry.set_defaults(func=cmd_telemetry)

    calib = subparsers.add_parser("calibrate", help="measure the pixel-to-ground transform")
    add_common(calib)
    calib.add_argument("--pairs", type=int, default=40)
    calib.add_argument("--out", type=Path, default=OUTPUTS / "calibration.json")
    calib.set_defaults(func=cmd_calibrate)

    detect = subparsers.add_parser("detect", help="find vehicles in each frame")
    add_common(detect, srt=False)
    add_frame_args(detect)
    add_detector_args(detect)
    detect.add_argument("--out", type=Path, default=OUTPUTS / "detections.csv")
    detect.set_defaults(func=cmd_detect)

    track = subparsers.add_parser("track", help="project detections and link them into tracks")
    add_common(track)
    track.add_argument("--detections", type=Path, default=OUTPUTS / "detections.csv")
    track.add_argument("--calibration", type=Path, default=OUTPUTS / "calibration.json")
    track.add_argument("--out", type=Path, default=OUTPUTS / "tracks.csv")
    track.set_defaults(func=cmd_track)

    draw = subparsers.add_parser("map", help="filter moving cars and draw the map")
    draw.add_argument("--tracks", type=Path, default=OUTPUTS / "tracks.csv")
    draw.add_argument("--srt", type=Path, default=DEFAULT_SRT)
    draw.add_argument("--out", type=Path, default=RESULTS / "map.html")
    draw.add_argument("--geojson", type=Path, default=RESULTS / "tracks.geojson")
    draw.set_defaults(func=cmd_map)

    run_all = subparsers.add_parser("all", help="run every stage")
    add_common(run_all)
    add_frame_args(run_all)
    add_detector_args(run_all)
    run_all.add_argument("--pairs", type=int, default=40)
    run_all.add_argument("--recalibrate", action="store_true")
    run_all.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
