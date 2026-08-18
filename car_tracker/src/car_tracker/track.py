
"""
Link per-frame detections into per-vehicle paths.

Association happens in **metres**, not pixels. In pixel space a parked car streaks
across the frame at the drone's speed while a car travelling alongside the drone barely
shifts, so pixel distance says nothing about whether two detections are the same
vehicle. In metres the test is physical: a car cannot cover more than
``max_speed * dt``.

Each track carries a constant-velocity Kalman filter over ``[east, north, v_east,
v_north]``. Velocity therefore falls out as a filter output rather than being
differenced from noisy positions, which matters because step 6 decides "moving versus
parked" from exactly that quantity.

Every detection is tracked, parked cars included. They surface as tracks that jitter in
place under GPS noise, and are separated later by
:mod:`car_tracker.postprocess`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from car_tracker.geo import latlon_to_metres, metres_to_latlon

# A car cannot plausibly exceed this on a city street; used to size the gate.
MAX_SPEED_MPS = 35.0

# Constant slack added to the gate: GPS noise dominates, plus the 0.29 m projection
# residual and centroid jitter from the detector.
GATE_SLACK_M = 4.0

# Hits before a track is trusted. Rejects single-frame false positives without
# needing a high detector confidence.
MIN_HITS = 3

# How long a track may survive on prediction alone. Time-based rather than
# frame-based so a strided run behaves the same as a full-rate one.
MAX_COAST_S = 1.0

# Kalman noise. Acceleration models a car changing speed; measurement noise reflects
# GPS plus projection error.
ACCEL_NOISE_MPS2 = 2.0
MEASUREMENT_NOISE_M = 2.0
INITIAL_SPEED_VARIANCE = 25.0

COLUMNS = (
    "track_id", "frame", "t_sec", "lat", "lon",
    "east_m", "north_m", "speed_mps", "heading_deg", "conf",
)


class TrackingError(RuntimeError):
    """
    Raised when tracking cannot proceed.
    """


class KalmanTrack:
    """
    Constant-velocity Kalman filter for one vehicle, in local ENU metres.

    Hand-rolled rather than pulled from a library: the model is four states and the
    matrices are small, so a dependency would add more surface than it saves.
    """

    def __init__(self, east: float, north: float, t_sec: float) -> None:
        self.state = np.array([east, north, 0.0, 0.0], dtype=float)
        self.covariance = np.diag(
            [
                MEASUREMENT_NOISE_M**2,
                MEASUREMENT_NOISE_M**2,
                INITIAL_SPEED_VARIANCE,
                INITIAL_SPEED_VARIANCE,
            ]
        )
        self.t_sec = t_sec

    @property
    def position(self) -> np.ndarray:
        return self.state[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[2:]

    @property
    def speed_mps(self) -> float:
        return float(np.hypot(*self.velocity))

    @property
    def heading_deg(self) -> float:
        """
        Direction of travel, degrees clockwise from north.
        """
        east, north = self.velocity
        return float(np.degrees(np.arctan2(east, north)) % 360.0)

    def predict(self, dt: float) -> np.ndarray:
        """
        Advance the state by ``dt`` seconds and return the predicted position.
        """
        if dt <= 0:
            return self.position.copy()

        transition = np.eye(4)
        transition[0, 2] = dt
        transition[1, 3] = dt

        # Discrete constant-acceleration process noise.
        variance = ACCEL_NOISE_MPS2**2
        dt2, dt3, dt4 = dt * dt, dt**3, dt**4
        block = np.array([[dt4 / 4.0, dt3 / 2.0], [dt3 / 2.0, dt2]]) * variance
        noise = np.zeros((4, 4))
        for axis in (0, 1):
            index = [axis, axis + 2]
            noise[np.ix_(index, index)] = block

        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + noise
        return self.position.copy()

    def seed_velocity(self, east: float, north: float, dt: float) -> None:
        """
        Initialise velocity from the first two observations.

        Without this the filter starts at rest and needs many frames to catch up, so
        reported speed is biased low for a track's whole early life — measured at 1.7x
        median understatement and up to 49x on short tracks. That bias was strong enough
        to reject real cars travelling 60 km/h as "too slow". Clamped to
        :data:`MAX_SPEED_MPS` so one noisy pair cannot launch the track.
        """
        if dt <= 0:
            return
        velocity = np.array([east - self.state[0], north - self.state[1]]) / dt
        speed = float(np.hypot(*velocity))
        if speed > MAX_SPEED_MPS:
            velocity *= MAX_SPEED_MPS / speed
        self.state[2:] = velocity

    def update(self, east: float, north: float) -> None:
        """
        Fold a measured position into the state.
        """
        observation = np.zeros((2, 4))
        observation[0, 0] = 1.0
        observation[1, 1] = 1.0
        measurement_noise = np.eye(2) * MEASUREMENT_NOISE_M**2

        residual = np.array([east, north]) - observation @ self.state
        residual_covariance = observation @ self.covariance @ observation.T + measurement_noise
        gain = self.covariance @ observation.T @ np.linalg.inv(residual_covariance)

        self.state = self.state + gain @ residual
        identity = np.eye(4)
        self.covariance = (identity - gain @ observation) @ self.covariance


@dataclass
class Track:
    """
    One vehicle's path under construction.

    Attributes:
        track_id: stable identifier.
        filter: motion model.
        hits: measurements absorbed.
        misses: consecutive frames without a match.
        last_t_sec: time of the most recent match, for coasting and dt.
        history: accepted observations, one record per matched frame.
    """

    track_id: int
    filter: KalmanTrack
    hits: int = 1
    misses: int = 0
    last_t_sec: float = 0.0
    history: list[dict] = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return self.hits >= MIN_HITS

    def record(self, frame: int, t_sec: float, origin: tuple[float, float], conf: float) -> None:
        """
        Append the current filtered state as an observation of this track.
        """
        east, north = self.filter.position
        lat, lon = metres_to_latlon(origin[0], origin[1], east, north)
        self.history.append(
            {
                "track_id": self.track_id,
                "frame": frame,
                "t_sec": t_sec,
                "lat": float(lat),
                "lon": float(lon),
                "east_m": float(east),
                "north_m": float(north),
                "speed_mps": self.filter.speed_mps,
                "heading_deg": self.filter.heading_deg,
                "conf": conf,
            }
        )


def to_local_metres(
    detections: pd.DataFrame, origin: tuple[float, float] | None = None
) -> tuple[pd.DataFrame, tuple[float, float]]:
    """
    Add ``east_m`` / ``north_m`` columns relative to a fixed origin.

    Args:
        detections: must contain ``lat`` and ``lon``.
        origin: reference point. Defaults to the first detection, which keeps
            coordinates small and comparable across the whole run.

    Returns:
        The augmented frame and the origin used.
    """
    for column in ("lat", "lon"):
        if column not in detections.columns:
            raise KeyError(f"detections is missing column {column!r}; project them first")
    if detections.empty:
        raise TrackingError("no detections to track")

    if origin is None:
        origin = (float(detections["lat"].iloc[0]), float(detections["lon"].iloc[0]))

    east, north = latlon_to_metres(
        origin[0], origin[1], detections["lat"].to_numpy(), detections["lon"].to_numpy()
    )
    result = detections.copy()
    result["east_m"] = east
    result["north_m"] = north
    return result, origin


def _assign(
    predictions: np.ndarray, measurements: np.ndarray, gate_m: float
) -> list[tuple[int, int]]:
    """
    Match predicted track positions to measurements, in metres.

    Uses Hungarian assignment for a globally optimal pairing; greedy nearest-neighbour
    mismatches cars that pass close to one another. Pairs beyond ``gate_m`` are
    dropped after solving, so the gate never forces a bad match.
    """
    if len(predictions) == 0 or len(measurements) == 0:
        return []

    cost = np.linalg.norm(predictions[:, None, :] - measurements[None, :, :], axis=2)
    # Make out-of-gate pairs unattractive rather than impossible, so the solver always
    # has a feasible problem.
    solvable = np.where(cost <= gate_m, cost, gate_m * 1e3)
    track_indices, detection_indices = linear_sum_assignment(solvable)
    return [
        (int(t), int(d))
        for t, d in zip(track_indices, detection_indices, strict=True)
        if cost[t, d] <= gate_m
    ]


def track_detections(
    detections: pd.DataFrame,
    origin: tuple[float, float] | None = None,
    max_speed_mps: float = MAX_SPEED_MPS,
    gate_slack_m: float = GATE_SLACK_M,
    max_coast_s: float = MAX_COAST_S,
    min_hits: int = MIN_HITS,
) -> pd.DataFrame:
    """
    Link projected detections into tracks.

    Args:
        detections: projected detections with ``frame``, ``t_sec``, ``lat``, ``lon``
            and optionally ``conf``.
        origin: ENU reference point; defaults to the first detection.
        max_speed_mps: fastest plausible vehicle, used to size the gate.
        gate_slack_m: constant added to the gate for measurement noise.
        max_coast_s: how long a track may survive without a match.
        min_hits: matches required before a track is emitted.

    Returns:
        One row per accepted observation, columns as in :data:`COLUMNS`, sorted by
        track then frame. Tracks never reaching ``min_hits`` are discarded.
    """
    for column in ("frame", "t_sec"):
        if column not in detections.columns:
            raise KeyError(f"detections is missing column {column!r}")

    localised, origin = to_local_metres(detections, origin)
    if "conf" not in localised.columns:
        localised["conf"] = np.nan

    identifiers = count(1)
    active: list[Track] = []
    finished: list[Track] = []

    for (frame, t_sec), group in localised.groupby(["frame", "t_sec"], sort=True):
        measurements = group[["east_m", "north_m"]].to_numpy(dtype=float)
        confidences = group["conf"].to_numpy(dtype=float)

        # Retire stale tracks before association, not after. Frames with no detections
        # never reach this loop, so a purely post-match check would let a track coast
        # across an arbitrarily long gap and then claim a far-away detection.
        surviving: list[Track] = []
        for track in active:
            if float(t_sec) - track.last_t_sec > max_coast_s:
                finished.append(track)
            else:
                surviving.append(track)
        active = surviving

        predictions = np.empty((len(active), 2))
        gate = gate_slack_m
        for index, track in enumerate(active):
            dt = float(t_sec) - track.last_t_sec
            predictions[index] = track.filter.predict(dt)
            gate = max(gate, max_speed_mps * dt + gate_slack_m)

        matches = _assign(predictions, measurements, gate)
        matched_tracks = {t for t, _ in matches}
        matched_detections = {d for _, d in matches}

        for track_index, detection_index in matches:
            track = active[track_index]
            east, north = measurements[detection_index]
            if track.hits == 1:
                track.filter.seed_velocity(east, north, float(t_sec) - track.last_t_sec)
            track.filter.update(east, north)
            track.hits += 1
            track.misses = 0
            track.last_t_sec = float(t_sec)
            track.record(int(frame), float(t_sec), origin, float(confidences[detection_index]))

        for index, track in enumerate(active):
            if index not in matched_tracks:
                track.misses += 1

        for detection_index in range(len(measurements)):
            if detection_index in matched_detections:
                continue
            east, north = measurements[detection_index]
            new_track = Track(
                track_id=next(identifiers),
                filter=KalmanTrack(east, north, float(t_sec)),
                last_t_sec=float(t_sec),
            )
            new_track.record(
                int(frame), float(t_sec), origin, float(confidences[detection_index])
            )
            active.append(new_track)

    finished.extend(active)
    rows = [row for track in finished if track.hits >= min_hits for row in track.history]
    if not rows:
        return pd.DataFrame(columns=list(COLUMNS))

    result = pd.DataFrame(rows, columns=list(COLUMNS))
    return result.sort_values(["track_id", "frame"], ignore_index=True)


def summarise(tracks: pd.DataFrame) -> pd.DataFrame:
    """
    One row per track: duration, straight-line displacement, path length and speed.

    ``displacement_m`` is what separates a moving car from a parked one: a parked car
    accumulates path length from noise but goes nowhere.
    """
    if tracks.empty:
        return pd.DataFrame(
            columns=[
                "track_id", "frames", "duration_s", "displacement_m",
                "path_length_m", "mean_speed_mps", "max_speed_mps",
            ]
        )

    rows = []
    for track_id, group in tracks.groupby("track_id", sort=True):
        east = group["east_m"].to_numpy()
        north = group["north_m"].to_numpy()
        steps = np.hypot(np.diff(east), np.diff(north))
        rows.append(
            {
                "track_id": int(track_id),
                "frames": len(group),
                "duration_s": float(group["t_sec"].iloc[-1] - group["t_sec"].iloc[0]),
                "displacement_m": float(np.hypot(east[-1] - east[0], north[-1] - north[0])),
                "path_length_m": float(steps.sum()),
                "mean_speed_mps": float(group["speed_mps"].mean()),
                "max_speed_mps": float(group["speed_mps"].max()),
            }
        )
    return pd.DataFrame(rows)
