"""
Measure the pixel-to-ground transform from the footage itself.

Two parameters cannot be trusted from the SRT (DECISIONS.md D5): the ground sample
distance, which the camera spec gets wrong by roughly a third, and the rotation that
maps image axes onto north, whose sign and zero-reference DJI does not document.

Both fall out of one observation. Between two frames the drone travels a known
distance and bearing according to GPS, while the static ground slides across the
image by a measurable number of pixels:

    GSD          = GPS distance / pixel shift magnitude
    rotation     = GPS bearing - direction of camera motion within the image

No sensor specification, object size or box accuracy is involved.

Accuracy is then self-checked without hand-picking a landmark: every tracked feature
is a static ground point, so projecting the same feature from both frames of a pair
must yield the same coordinate. The distance between those two projections, over
hundreds of features, is the residual reported as ``Calibration.residual_m``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from car_tracker.geo import Calibration, CameraModel, bearing_deg, haversine_m
from car_tracker.video import read_frame, video_capture

# Feature tracking. Corners are requested generously because winter fields and
# asphalt are low-texture; RANSAC discards whatever does not fit the dominant motion.
MAX_CORNERS = 2000
QUALITY_LEVEL = 0.01
MIN_CORNER_DISTANCE = 8
LK_WINDOW = (21, 21)
LK_PYRAMID_LEVELS = 3
RANSAC_THRESHOLD_PX = 2.0

# Minimum tracked points for a pair to be trusted at all.
MIN_INLIERS = 30

# Pair selection. A pair is only usable when the drone translates enough to measure
# but the camera geometry barely changes, so the image motion is near-pure
# translation and scale is constant across the pair.
DEFAULT_GAP = 15  # frames apart, ~0.5 s at 30 fps
MIN_GPS_DISTANCE_M = 3.0
MAX_YAW_DELTA_DEG = 1.0
MAX_ALT_DELTA_M = 0.5


class CalibrationError(RuntimeError):
    """
    Raised when calibration cannot be measured from the available frames.
    """


def wrap180(degrees: np.ndarray | float) -> np.ndarray:
    """
    Wrap angles into [-180, 180).
    """
    return (np.asarray(degrees, dtype=float) + 180.0) % 360.0 - 180.0


def circular_mean_deg(degrees: np.ndarray) -> float:
    """
    Mean of angles, correct across the +-180 discontinuity.
    """
    radians = np.radians(np.asarray(degrees, dtype=float))
    return float(np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())))


@dataclass(frozen=True)
class ShiftMeasurement:
    """
    Apparent motion of the static ground between two frames.

    Attributes:
        dx, dy: shift of the image centre in pixels, from the first frame to the
            second. Positive ``dy`` is downward, matching image convention.
        rotation_deg: image rotation implied by the fitted transform.
        scale: fitted scale change, ~1.0 when altitude is constant.
        inliers: tracked points consistent with the dominant motion.
        tracked: points successfully followed by optical flow.
        points_a, points_b: inlier correspondences, used for the residual check.
    """

    dx: float
    dy: float
    rotation_deg: float
    scale: float
    inliers: int
    tracked: int
    points_a: np.ndarray
    points_b: np.ndarray

    @property
    def magnitude_px(self) -> float:
        return float(np.hypot(self.dx, self.dy))


def measure_shift(image_a: np.ndarray, image_b: np.ndarray) -> ShiftMeasurement:
    """
    Estimate how the static ground moved between two frames.

    Tracks corners with Lucas-Kanade optical flow, then fits a similarity transform
    with RANSAC. Cars and other independently moving objects fail to fit the dominant
    ground motion and are rejected as outliers, which is why no masking is needed.

    Raises:
        CalibrationError: if too few points survive tracking or RANSAC.
    """
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(
        gray_a,
        maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_CORNER_DISTANCE,
    )
    if corners is None or len(corners) < MIN_INLIERS:
        raise CalibrationError(f"only {0 if corners is None else len(corners)} corners found")

    moved, status, _ = cv2.calcOpticalFlowPyrLK(
        gray_a, gray_b, corners, None, winSize=LK_WINDOW, maxLevel=LK_PYRAMID_LEVELS
    )
    keep = status.ravel() == 1
    points_a = corners[keep].reshape(-1, 2)
    points_b = moved[keep].reshape(-1, 2)
    if len(points_a) < MIN_INLIERS:
        raise CalibrationError(f"only {len(points_a)} points tracked")

    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        points_a, points_b, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD_PX
    )
    if matrix is None:
        raise CalibrationError("could not fit a similarity transform")

    inliers = inlier_mask.ravel().astype(bool)
    if inliers.sum() < MIN_INLIERS:
        raise CalibrationError(f"only {int(inliers.sum())} RANSAC inliers")

    # Report the shift of the image centre rather than the raw translation column:
    # with any rotation present, the translation of the origin is not the motion the
    # centre of the scene actually underwent.
    height, width = gray_a.shape
    centre = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
    moved_centre = matrix @ np.array([centre[0], centre[1], 1.0])

    return ShiftMeasurement(
        dx=float(moved_centre[0] - centre[0]),
        dy=float(moved_centre[1] - centre[1]),
        rotation_deg=float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
        scale=float(np.hypot(matrix[0, 0], matrix[1, 0])),
        inliers=int(inliers.sum()),
        tracked=int(len(points_a)),
        points_a=points_a[inliers],
        points_b=points_b[inliers],
    )


def select_pairs(
    telemetry: pd.DataFrame,
    gap: int = DEFAULT_GAP,
    count: int = 40,
    min_distance_m: float = MIN_GPS_DISTANCE_M,
    max_yaw_delta_deg: float = MAX_YAW_DELTA_DEG,
    max_alt_delta_m: float = MAX_ALT_DELTA_M,
) -> pd.DataFrame:
    """
    Choose frame pairs suitable for measuring the transform.

    A usable pair has the drone translating far enough for the pixel shift to be
    measured precisely, while yaw and altitude stay put so the image motion is
    near-pure translation at constant scale.

    Returns:
        Columns ``frame_a``, ``frame_b``, ``gps_distance_m``, ``gps_bearing_deg``,
        ``yaw``, ``gsd_spec``, spread evenly across the flight.

    Raises:
        CalibrationError: if no pair meets the criteria.
    """
    first = telemetry.iloc[:-gap].reset_index(drop=True)
    second = telemetry.iloc[gap:].reset_index(drop=True)

    lat_a, lon_a = first["lat"], first["lon"]
    lat_b, lon_b = second["lat"], second["lon"]

    pairs = pd.DataFrame(
        {
            "frame_a": first["frame"],
            "frame_b": second["frame"],
            "gps_distance_m": haversine_m(lat_a, lon_a, lat_b, lon_b),
            "gps_bearing_deg": bearing_deg(lat_a, lon_a, lat_b, lon_b),
            "yaw": first["yaw"],
            "yaw_delta": np.abs(wrap180(second["yaw"] - first["yaw"])),
            "alt_delta": (second["rel_alt"] - first["rel_alt"]).abs(),
            "gsd_spec": first["gsd"],
        }
    )
    usable = pairs[
        (pairs["gps_distance_m"] >= min_distance_m)
        & (pairs["yaw_delta"] <= max_yaw_delta_deg)
        & (pairs["alt_delta"] <= max_alt_delta_m)
    ]
    if usable.empty:
        raise CalibrationError(
            "no frame pair satisfies the selection criteria; "
            f"best distance {pairs['gps_distance_m'].max():.1f} m, "
            f"smallest yaw delta {pairs['yaw_delta'].min():.2f} deg"
        )

    # Spread the sample over the whole flight so one stretch of terrain cannot
    # dominate, and so the yaw fit sees a range of headings.
    step = max(1, len(usable) // count)
    return usable.iloc[::step].head(count).reset_index(drop=True)


def measure_pairs(
    video_path: str | Path, pairs: pd.DataFrame, image_size: tuple[int, int]
) -> pd.DataFrame:
    """
    Measure ground motion for each candidate pair.

    Failed pairs are skipped with their reason recorded rather than aborting the run,
    since low-texture frames are expected over snow and bare fields.

    Returns:
        One row per pair with the measured shift, implied GSD and implied rotation.
    """
    width, height = image_size
    centre = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
    rows: list[dict] = []

    with video_capture(video_path) as capture:
        for pair in pairs.itertuples(index=False):
            try:
                shift = measure_shift(
                    read_frame(capture, int(pair.frame_a)),
                    read_frame(capture, int(pair.frame_b)),
                )
            except (CalibrationError, cv2.error) as error:
                rows.append(
                    {"frame_a": pair.frame_a, "frame_b": pair.frame_b, "error": str(error)}
                )
                continue

            # The ground slides one way, so the camera travelled the other. Express
            # that camera motion as an angle clockwise from image-up.
            camera_right = -shift.dx
            camera_up = shift.dy
            heading_in_image = np.degrees(np.arctan2(camera_right, camera_up))

            gsd_measured = pair.gps_distance_m / shift.magnitude_px
            rows.append(
                {
                    "frame_a": pair.frame_a,
                    "frame_b": pair.frame_b,
                    "yaw": pair.yaw,
                    "gps_distance_m": pair.gps_distance_m,
                    "gps_bearing_deg": pair.gps_bearing_deg,
                    "shift_px": shift.magnitude_px,
                    "dx": shift.dx,
                    "dy": shift.dy,
                    "image_rotation_deg": shift.rotation_deg,
                    "scale": shift.scale,
                    "inliers": shift.inliers,
                    "tracked": shift.tracked,
                    "gsd_spec": pair.gsd_spec,
                    "gsd_measured": gsd_measured,
                    "gsd_scale": gsd_measured / pair.gsd_spec,
                    # Bearing of image-up: the rotation CameraModel needs.
                    "rotation_deg": wrap180(pair.gps_bearing_deg - heading_in_image),
                    "points_a": shift.points_a - centre,
                    "points_b": shift.points_b - centre,
                    "error": None,
                }
            )
    return pd.DataFrame(rows)


def fit_yaw_convention(measurements: pd.DataFrame) -> tuple[int, float, float]:
    """
    Find which signed telemetry yaw matches the measured image-up bearing.

    Tries both sign conventions, taking the constant offset as the circular mean of
    the residual, and keeps whichever fits better.

    Returns:
        ``(yaw_sign, yaw_offset_deg, spread_deg)`` where the spread is the standard
        deviation of the residual and indicates how consistent the fit is.
    """
    rotation = measurements["rotation_deg"].to_numpy()
    yaw = measurements["yaw"].to_numpy()

    best: tuple[int, float, float] | None = None
    for sign in (1, -1):
        residual = wrap180(rotation - sign * yaw)
        offset = circular_mean_deg(residual)
        spread = float(np.std(wrap180(residual - offset)))
        if best is None or spread < best[2]:
            best = (sign, offset, spread)
    assert best is not None
    return best


def static_point_residual_m(
    measurements: pd.DataFrame, telemetry: pd.DataFrame, model: CameraModel
) -> pd.Series:
    """
    How far a static ground point moves when projected from both frames of a pair.

    Every RANSAC inlier is a static ground feature, so a correct transform projects
    it to the same coordinate from either frame. This replaces hand-picking a
    landmark and yields hundreds of samples instead of one.

    Returns:
        Median drift in metres, per pair.
    """
    indexed = telemetry.set_index("frame")
    centre_u, centre_v = model.centre
    drifts: list[float] = []

    for row in measurements.itertuples(index=False):
        rows_a = indexed.loc[row.frame_a]
        rows_b = indexed.loc[row.frame_b]
        offsets_a, offsets_b = row.points_a, row.points_b

        lat_a, lon_a = model.pixel_to_latlon(
            offsets_a[:, 0] + centre_u, offsets_a[:, 1] + centre_v,
            rows_a["lat"], rows_a["lon"], rows_a["gsd"], rows_a["yaw"],
        )
        lat_b, lon_b = model.pixel_to_latlon(
            offsets_b[:, 0] + centre_u, offsets_b[:, 1] + centre_v,
            rows_b["lat"], rows_b["lon"], rows_b["gsd"], rows_b["yaw"],
        )
        drifts.append(float(np.median(haversine_m(lat_a, lon_a, lat_b, lon_b))))

    return pd.Series(drifts, name="residual_m")


@dataclass
class CalibrationResult:
    """
    Outcome of a calibration run, including the evidence behind it.
    """

    calibration: Calibration
    measurements: pd.DataFrame
    yaw_spread_deg: float

    @property
    def used(self) -> pd.DataFrame:
        return self.measurements[self.measurements["error"].isna()]

    def report(self) -> str:
        cal = self.calibration
        used = self.used
        return "\n".join(
            [
                f"pairs measured   {len(used)} of {len(self.measurements)}",
                f"inliers/pair     median {used['inliers'].median():.0f}",
                f"shift            median {used['shift_px'].median():.1f} px",
                f"gsd_spec         median {used['gsd_spec'].median() * 100:.2f} cm/px",
                f"gsd_measured     median {used['gsd_measured'].median() * 100:.2f} cm/px",
                f"gsd_scale        {cal.gsd_scale:.3f}"
                f"  (IQR {used['gsd_scale'].quantile(0.25):.3f}"
                f"..{used['gsd_scale'].quantile(0.75):.3f})",
                f"yaw convention   rotation = {cal.yaw_sign:+d} * yaw "
                f"{cal.yaw_offset_deg:+.2f} deg   (spread {self.yaw_spread_deg:.2f} deg)",
                f"residual         {cal.residual_m:.2f} m"
                if cal.residual_m is not None
                else "residual         not evaluated",
            ]
        )


def calibrate(
    video_path: str | Path,
    telemetry: pd.DataFrame,
    image_size: tuple[int, int] = (1920, 1080),
    gap: int = DEFAULT_GAP,
    count: int = 40,
) -> CalibrationResult:
    """
    Measure the pixel-to-ground transform and its accuracy.

    Args:
        video_path: source footage.
        telemetry: table from :mod:`car_tracker.telemetry`.
        image_size: (width, height) in pixels.
        gap: frames between the two halves of a pair.
        count: how many pairs to measure.

    Raises:
        CalibrationError: if no pair could be measured.
    """
    pairs = select_pairs(telemetry, gap=gap, count=count)
    measurements = measure_pairs(video_path, pairs, image_size)
    used = measurements[measurements["error"].isna()]
    if used.empty:
        reasons = measurements["error"].dropna().unique()[:3]
        raise CalibrationError(f"no pair could be measured; reasons: {list(reasons)}")

    # Median over pairs, not mean: a single mistracked pair would otherwise drag the
    # scale with it.
    gsd_scale = float(used["gsd_scale"].median())
    yaw_sign, yaw_offset, yaw_spread = fit_yaw_convention(used)

    calibration = Calibration(
        gsd_scale=gsd_scale, yaw_offset_deg=yaw_offset, yaw_sign=yaw_sign
    )
    model = CameraModel(image_size=image_size, calibration=calibration)
    residual = static_point_residual_m(used, telemetry, model)

    return CalibrationResult(
        calibration=Calibration(
            gsd_scale=gsd_scale,
            yaw_offset_deg=yaw_offset,
            yaw_sign=yaw_sign,
            residual_m=float(residual.median()),
        ),
        measurements=measurements,
        yaw_spread_deg=yaw_spread,
    )
