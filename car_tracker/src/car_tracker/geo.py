"""
Convert image pixels to geographic coordinates for a nadir camera.

The gimbal points straight down for the whole flight (pitch is -90 deg in every
frame), so the mapping from pixel to ground is a similarity transform rather than a
perspective projection:

1. offset the pixel from the image centre
2. scale by the ground sample distance to get metres
3. rotate so image axes align with north/east
4. add to the drone's own position

Two parameters of that transform cannot be trusted from the SRT and are measured
instead by :mod:`car_tracker.calibrate` (see DECISIONS.md D5):

``gsd_scale``
    Multiplier on the spec-derived GSD. Measured cars appear ~30% smaller than the
    24 mm-equivalent spec predicts, so the nominal value is known to be wrong.

``yaw_offset`` / ``yaw_sign``
    The rotation that maps image axes onto north. DJI's yaw sign and zero-reference
    are undocumented, so the rotation is fitted rather than assumed.

Displacements are expressed in East/North metres and resolved on the WGS84 ellipsoid
via :mod:`pyproj`, so distance, bearing and offset all share one Earth model. Working
in metres is what makes later association thresholds physically meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyproj import Geod

_GEOD = Geod(ellps="WGS84")

DEFAULT_IMAGE_SIZE = (1920, 1080)


def metres_to_latlon(
    lat0: np.ndarray | float,
    lon0: np.ndarray | float,
    east_m: np.ndarray | float,
    north_m: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Offset a coordinate by an east/north displacement in metres.

    Solved as a geodesic on WGS84 rather than with a spherical approximation, so it
    is exactly consistent with :func:`haversine_m`. A spherical radius disagrees with
    the ellipsoid by ~0.1% at this latitude, which would silently bias every
    projected coordinate.
    """
    lat0_a, lon0_a, east, north = np.broadcast_arrays(
        np.asarray(lat0, dtype=float),
        np.asarray(lon0, dtype=float),
        np.asarray(east_m, dtype=float),
        np.asarray(north_m, dtype=float),
    )
    azimuth = np.degrees(np.arctan2(east, north))
    distance = np.hypot(east, north)
    lon, lat, _ = _GEOD.fwd(lon0_a, lat0_a, azimuth, distance)
    return np.asarray(lat), np.asarray(lon)


def latlon_to_metres(
    lat0: np.ndarray | float,
    lon0: np.ndarray | float,
    lat: np.ndarray | float,
    lon: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    East/north displacement in metres from ``(lat0, lon0)`` to ``(lat, lon)``.

    Inverse of :func:`metres_to_latlon`.
    """
    azimuth, _, distance = _GEOD.inv(
        np.asarray(lon0, dtype=float),
        np.asarray(lat0, dtype=float),
        np.asarray(lon, dtype=float),
        np.asarray(lat, dtype=float),
    )
    radians = np.radians(azimuth)
    return np.asarray(distance) * np.sin(radians), np.asarray(distance) * np.cos(radians)


def haversine_m(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray:
    """
    Ground distance in metres between coordinate pairs, on the WGS84 ellipsoid.
    """
    _, _, dist = _GEOD.inv(
        np.asarray(lon1, dtype=float),
        np.asarray(lat1, dtype=float),
        np.asarray(lon2, dtype=float),
        np.asarray(lat2, dtype=float),
    )
    return np.asarray(dist, dtype=float)


def bearing_deg(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray:
    """
    Forward azimuth in degrees clockwise from true north, in [0, 360).
    """
    az, _, _ = _GEOD.inv(
        np.asarray(lon1, dtype=float),
        np.asarray(lat1, dtype=float),
        np.asarray(lon2, dtype=float),
        np.asarray(lat2, dtype=float),
    )
    return np.mod(np.asarray(az, dtype=float), 360.0)


@dataclass(frozen=True)
class Calibration:
    """
    The measured parameters of the pixel-to-ground transform.

    Attributes:
        gsd_scale: multiplier applied to the spec-derived GSD. 1.0 means the spec is
            correct; measurement so far suggests ~1.3.
        yaw_offset_deg: constant added to the telemetry yaw to obtain the rotation
            from image axes to north.
        yaw_sign: +1 or -1, whichever direction of telemetry yaw matches reality.
        residual_m: how far a static point drifts under this calibration. The
            accuracy figure; ``None`` until validated.
    """

    gsd_scale: float = 1.0
    yaw_offset_deg: float = 0.0
    yaw_sign: int = 1
    residual_m: float | None = None

    def __post_init__(self) -> None:
        if self.yaw_sign not in (1, -1):
            raise ValueError(f"yaw_sign must be +1 or -1, got {self.yaw_sign}")
        if self.gsd_scale <= 0:
            raise ValueError(f"gsd_scale must be positive, got {self.gsd_scale}")


class CameraModel:
    """
    Projects image pixels onto the ground for a nadir camera.

    Args:
        image_size: (width, height) in pixels.
        calibration: measured transform parameters. Defaults are the uncalibrated
            spec values, which are known to be wrong — see DECISIONS.md D5.

    Example:
        >>> model = CameraModel()
        >>> lat, lon = model.pixel_to_latlon(960, 540, drone_lat=48.267,
        ...                                  drone_lon=25.914, gsd=0.08, yaw_deg=-65.8)
        >>> float(lat), float(lon)   # image centre maps to the drone's own position
        (48.267, 25.914)
    """

    def __init__(
        self,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        calibration: Calibration | None = None,
    ) -> None:
        self.width, self.height = image_size
        self.calibration = calibration or Calibration()

    @property
    def centre(self) -> tuple[float, float]:
        """
        Principal point, assumed to be the image centre (no distortion data).
        """
        return (self.width - 1) / 2.0, (self.height - 1) / 2.0

    def rotation_deg(self, yaw_deg: np.ndarray | float) -> np.ndarray:
        """
        Rotation from image axes to north, in degrees, for the given telemetry yaw.
        """
        cal = self.calibration
        return cal.yaw_sign * np.asarray(yaw_deg, dtype=float) + cal.yaw_offset_deg

    def pixel_offset_metres(
        self,
        u: np.ndarray | float,
        v: np.ndarray | float,
        gsd: np.ndarray | float,
        yaw_deg: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Ground offset in metres (east, north) of a pixel from the image centre.

        Image ``v`` grows downward while north grows upward, so the vertical axis is
        negated before rotation.
        """
        cx, cy = self.centre
        scaled_gsd = np.asarray(gsd, dtype=float) * self.calibration.gsd_scale

        right_m = (np.asarray(u, dtype=float) - cx) * scaled_gsd
        up_m = (cy - np.asarray(v, dtype=float)) * scaled_gsd

        theta = np.radians(self.rotation_deg(yaw_deg))
        sin, cos = np.sin(theta), np.cos(theta)

        # Rotate the image frame so "up" points along the camera heading: a heading
        # of 0 deg leaves up pointing north, 90 deg swings it to east.
        east_m = up_m * sin + right_m * cos
        north_m = up_m * cos - right_m * sin
        return east_m, north_m

    def pixel_to_latlon(
        self,
        u: np.ndarray | float,
        v: np.ndarray | float,
        drone_lat: np.ndarray | float,
        drone_lon: np.ndarray | float,
        gsd: np.ndarray | float,
        yaw_deg: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Ground coordinate beneath an image pixel.

        All arguments broadcast, so a whole detection table can be projected in one
        call rather than row by row.
        """
        east_m, north_m = self.pixel_offset_metres(u, v, gsd, yaw_deg)
        return metres_to_latlon(drone_lat, drone_lon, east_m, north_m)

    def project_detections(
        self,
        detections: pd.DataFrame,
        telemetry: pd.DataFrame,
        u_col: str = "u",
        v_col: str = "v",
    ) -> pd.DataFrame:
        """
        Attach ``lat``/``lon`` to a detection table by joining telemetry on ``frame``.

        Args:
            detections: must contain ``frame`` and the pixel columns.
            telemetry: table from :mod:`car_tracker.telemetry`.

        Returns:
            A copy of ``detections`` with ``lat``, ``lon`` and the telemetry columns
            used for the projection.

        Raises:
            KeyError: if required columns are missing.
            ValueError: if any detection references an unknown frame.
        """
        for column in ("frame", u_col, v_col):
            if column not in detections.columns:
                raise KeyError(f"detections is missing column {column!r}")

        needed = ["frame", "lat", "lon", "rel_alt", "gsd", "yaw", "t_sec"]
        merged = detections.merge(
            telemetry[needed].rename(columns={"lat": "drone_lat", "lon": "drone_lon"}),
            on="frame",
            how="left",
            validate="many_to_one",
        )
        if merged["drone_lat"].isna().any():
            missing = merged.loc[merged["drone_lat"].isna(), "frame"].unique()[:10]
            raise ValueError(f"no telemetry for frames {list(missing)}")

        merged["lat"], merged["lon"] = self.pixel_to_latlon(
            merged[u_col].to_numpy(),
            merged[v_col].to_numpy(),
            merged["drone_lat"].to_numpy(),
            merged["drone_lon"].to_numpy(),
            merged["gsd"].to_numpy(),
            merged["yaw"].to_numpy(),
        )
        return merged

    def footprint_m(self, gsd: float) -> tuple[float, float]:
        """
        Ground size (width, height) covered by the whole frame, in metres.
        """
        scaled = gsd * self.calibration.gsd_scale
        return self.width * scaled, self.height * scaled

    def __repr__(self) -> str:
        cal = self.calibration
        return (
            f"CameraModel({self.width}x{self.height}, gsd_scale={cal.gsd_scale:.3f}, "
            f"yaw={cal.yaw_sign:+d}*yaw{cal.yaw_offset_deg:+.1f}deg)"
        )
