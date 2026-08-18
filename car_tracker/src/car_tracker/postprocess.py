
"""
Separate moving vehicles from parked ones, and export the survivors.

This is where the assignment is actually satisfied. The detector finds every vehicle,
and on this footage most of them are parked — a snowy street with roughly twenty
stationary cars per frame and a couple driving. A map of all detections would answer
the wrong question.

The discriminator is geometric, not visual. Once tracks are geo-referenced, a parked
car holds a constant coordinate while the drone flies over it, and a moving car
translates. Three signals separate them:

``displacement``
    Straight-line distance from the first observation to the last. Near zero for a
    parked car whatever the noise does.

``straightness``
    Displacement divided by accumulated path length. Noise wanders, so a parked car
    accumulates path length while going nowhere, giving a ratio near zero. A vehicle
    driving down a road approaches one.

``speed``
    Median filtered speed, which the Kalman filter already provides.

A correction is applied first: drift in the drone's own GPS fix is common-mode, so it
displaces every projected object together. Left alone it makes whole streets of parked
cars appear to drift in formation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# A vehicle must cover this much ground before it counts as moving. Comfortably above
# GPS noise and the 0.29 m projection residual, comfortably below what a car in
# traffic covers in a few seconds.
MIN_DISPLACEMENT_M = 15.0

# Displacement / path length. Rejects tracks that accumulate distance by jittering.
MIN_STRAIGHTNESS = 0.4

# Average speed over the track, computed as displacement / duration. Deliberately not
# the Kalman filter's own speed: that starts at rest and is biased low by its warm-up
# (1.7x median, up to 49x on short tracks), which rejected real cars doing 60 km/h as
# "too slow". Displacement over duration needs no filter convergence.
#
# The floor is 5 m/s (18 km/h) rather than something near zero because of a measured
# artifact: during sustained fast rotation (~98 deg/s) parked cars creep 20-28 m over
# 5-7 s, i.e. ~14 km/h. That is too slow for a car genuinely driving on a street and too
# fast to be parked, so the floor separates them. Cost: a vehicle crawling in a jam below
# 18 km/h is not reported.
MIN_AVERAGE_SPEED_MPS = 5.0

# Tracks shorter than this are too brief to judge, and are usually detector noise.
MIN_DURATION_S = 0.5
MIN_OBSERVATIONS = 5

# Above this, a "vehicle" is an association error rather than a car.
MAX_PLAUSIBLE_SPEED_MPS = 45.0

# Circular spread of the per-step heading, in degrees. A vehicle on a road holds its
# direction or turns smoothly; a projection artifact wanders. Measured on real tracks: a
# straight run scores under 10 deg, a 90 deg turn about 26, a U-turn about 52, while
# erratic false movers reach 60-108. The limit therefore permits ~140 deg of genuine
# turning and still rejects paths that visibly cannot be driven.
#
# This catches what straightness misses: straightness only compares endpoints to path
# length, so a zigzag ending up far away still scores 1.0.
MAX_HEADING_SPREAD_DEG = 40.0

# Steps shorter than this carry no reliable direction, so they are excluded from the
# heading statistic rather than contributing noise.
MIN_STEP_FOR_HEADING_M = 0.05
MIN_STEPS_FOR_HEADING = 5

# Window for smoothing positions, in observations. Odd so it is symmetric.
SMOOTHING_WINDOW = 5


@dataclass(frozen=True)
class MovingCriteria:
    """
    Thresholds separating moving vehicles from parked ones.
    """

    min_displacement_m: float = MIN_DISPLACEMENT_M
    min_straightness: float = MIN_STRAIGHTNESS
    min_average_speed_mps: float = MIN_AVERAGE_SPEED_MPS
    min_duration_s: float = MIN_DURATION_S
    min_observations: int = MIN_OBSERVATIONS
    max_speed_mps: float = MAX_PLAUSIBLE_SPEED_MPS
    max_heading_spread_deg: float = MAX_HEADING_SPREAD_DEG


def estimate_gps_drift(tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate common-mode displacement per frame from the population of tracks.

    Drift in the drone's GPS fix shifts every projected object by the same amount, so
    the median step across all tracks present in a frame approximates it. Taking the
    median rather than the mean means the handful of genuinely moving vehicles do not
    bias the estimate, provided most tracks are parked — which is the case here.

    Returns:
        Columns ``t_sec``, ``drift_east_m``, ``drift_north_m``: the cumulative drift
        to subtract at each timestamp.
    """
    if tracks.empty:
        return pd.DataFrame(columns=["t_sec", "drift_east_m", "drift_north_m"])

    ordered = tracks.sort_values(["track_id", "t_sec"])
    steps = ordered.groupby("track_id", sort=False)[["east_m", "north_m"]].diff()
    steps["t_sec"] = ordered["t_sec"]

    per_time = (
        steps.dropna()
        .groupby("t_sec", sort=True)[["east_m", "north_m"]]
        .median()
        .rename(columns={"east_m": "drift_east_m", "north_m": "drift_north_m"})
    )
    cumulative = per_time.cumsum().reset_index()

    # The first timestamp has no step of its own, so drift there is zero by definition.
    first = pd.DataFrame(
        {"t_sec": [tracks["t_sec"].min()], "drift_east_m": [0.0], "drift_north_m": [0.0]}
    )
    return pd.concat([first, cumulative], ignore_index=True).drop_duplicates(
        "t_sec", keep="last"
    )


def remove_gps_drift(tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Subtract the common-mode drift estimated by :func:`estimate_gps_drift`.

    Adds ``east_corrected_m`` / ``north_corrected_m``, leaving the originals intact so
    the correction can be inspected rather than taken on trust.
    """
    if tracks.empty:
        result = tracks.copy()
        result["east_corrected_m"] = []
        result["north_corrected_m"] = []
        return result

    drift = estimate_gps_drift(tracks)
    merged = tracks.merge(drift, on="t_sec", how="left")
    merged[["drift_east_m", "drift_north_m"]] = (
        merged[["drift_east_m", "drift_north_m"]].ffill().fillna(0.0)
    )
    merged["east_corrected_m"] = merged["east_m"] - merged["drift_east_m"]
    merged["north_corrected_m"] = merged["north_m"] - merged["drift_north_m"]
    return merged


def smooth_positions(
    tracks: pd.DataFrame,
    window: int = SMOOTHING_WINDOW,
    east_col: str = "east_m",
    north_col: str = "north_m",
) -> pd.DataFrame:
    """
    Rolling-median smoothing of each track's position.

    A median rather than a mean, so a single badly-placed detection is rejected
    outright instead of being averaged into the path.

    Args:
        east_col, north_col: source columns. Named explicitly because drift correction
            writes to ``*_corrected_m``; smoothing the raw columns regardless silently
            discarded that correction.
    """
    if tracks.empty:
        return tracks.copy()

    result = tracks.sort_values(["track_id", "t_sec"]).copy()
    for source, target in ((east_col, "east_smooth_m"), (north_col, "north_smooth_m")):
        result[target] = result.groupby("track_id", sort=False)[source].transform(
            lambda values: values.rolling(window, center=True, min_periods=1).median()
        )
    return result


def heading_spread_deg(east: np.ndarray, north: np.ndarray) -> float:
    """
    Circular standard deviation of the per-step direction of travel, in degrees.

    Zero means every step pointed the same way. Large values mean the path doubles back on
    itself, which a car on a road does not do.

    Returns:
        0.0 when there are too few moving steps to judge, so short or stationary tracks
        are not penalised by this test.
    """
    de, dn = np.diff(east), np.diff(north)
    steps = np.hypot(de, dn)
    usable = steps > MIN_STEP_FOR_HEADING_M
    if usable.sum() < MIN_STEPS_FOR_HEADING:
        return 0.0

    angles = np.arctan2(de[usable], dn[usable])
    resultant = np.abs(np.mean(np.exp(1j * angles)))
    return float(np.degrees(np.sqrt(max(-2.0 * np.log(max(resultant, 1e-9)), 0.0))))


def track_features(
    tracks: pd.DataFrame, east_col: str = "east_m", north_col: str = "north_m"
) -> pd.DataFrame:
    """
    Per-track geometry used to classify motion.

    Returns:
        One row per track with observation count, duration, displacement, path length,
        straightness, speeds and endpoint coordinates.
    """
    columns = [
        "track_id", "observations", "duration_s", "displacement_m", "path_length_m",
        "straightness", "heading_spread_deg", "average_speed_mps", "median_speed_mps",
        "max_speed_mps", "mean_conf",
        "start_lat", "start_lon", "end_lat", "end_lon",
    ]
    if tracks.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for track_id, group in tracks.sort_values("t_sec").groupby("track_id", sort=True):
        east = group[east_col].to_numpy(dtype=float)
        north = group[north_col].to_numpy(dtype=float)
        path_length = float(np.hypot(np.diff(east), np.diff(north)).sum())
        displacement = float(np.hypot(east[-1] - east[0], north[-1] - north[0]))
        rows.append(
            {
                "track_id": int(track_id),
                "observations": len(group),
                "duration_s": float(group["t_sec"].iloc[-1] - group["t_sec"].iloc[0]),
                "displacement_m": displacement,
                "path_length_m": path_length,
                # A degenerate zero-length path is stationary by definition, not
                # perfectly straight.
                "straightness": float(displacement / path_length) if path_length > 0 else 0.0,
                "heading_spread_deg": heading_spread_deg(east, north),
                # Ground truth for "did this thing move": no filter involved.
                "average_speed_mps": (
                    float(displacement / duration) if (duration := float(
                        group["t_sec"].iloc[-1] - group["t_sec"].iloc[0]
                    )) > 0 else 0.0
                ),
                "median_speed_mps": float(group["speed_mps"].median()),
                "max_speed_mps": float(group["speed_mps"].max()),
                "mean_conf": float(group["conf"].mean()) if "conf" in group else float("nan"),
                "start_lat": float(group["lat"].iloc[0]),
                "start_lon": float(group["lon"].iloc[0]),
                "end_lat": float(group["lat"].iloc[-1]),
                "end_lon": float(group["lon"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def classify(
    features: pd.DataFrame, criteria: MovingCriteria | None = None
) -> pd.DataFrame:
    """
    Label each track as moving, parked, or too short to judge.

    Adds boolean ``is_moving`` plus a human-readable ``verdict`` naming the first
    failed test, so a rejected track can be argued with rather than merely dropped.
    """
    criteria = criteria or MovingCriteria()
    if features.empty:
        result = features.copy()
        result["is_moving"] = []
        result["verdict"] = []
        return result

    result = features.copy()
    too_short = (result["observations"] < criteria.min_observations) | (
        result["duration_s"] < criteria.min_duration_s
    )
    implausible = result["max_speed_mps"] > criteria.max_speed_mps
    still = result["displacement_m"] < criteria.min_displacement_m
    wandering = result["straightness"] < criteria.min_straightness
    erratic = result.get(
        "heading_spread_deg", pd.Series(0.0, index=result.index)
    ) > criteria.max_heading_spread_deg
    slow = result["average_speed_mps"] < criteria.min_average_speed_mps

    result["is_moving"] = ~(too_short | implausible | still | wandering | erratic | slow)
    # Displacement-based tests come first because they are the most robust signal.
    # Speed can spike from a single mis-placed detection, so a track that plainly went
    # nowhere should read "stationary" rather than "implausible_speed". A genuine
    # association error has large displacement and falls through to the speed test.
    result["verdict"] = np.select(
        [too_short, still, wandering, erratic, slow, implausible],
        ["too_short", "stationary", "wandering", "erratic", "too_slow", "implausible_speed"],
        default="moving",
    )
    return result


def moving_tracks(
    tracks: pd.DataFrame,
    criteria: MovingCriteria | None = None,
    correct_drift: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full post-processing pass.

    Applies smoothing, measures each track, then classifies it.

    Drift correction is **off by default**. It addresses a real phenomenon — the drone's
    own GPS wander displaces every projected object together — but the estimator sums a
    per-frame median, and the small bias in that median integrates into unbounded false
    drift: on this footage it reports 117 m east and -108 m north where the truth is a few
    metres. Left on, it cancels genuine motion when several vehicles move together. Enable
    it only with evidence of real drift, and check the estimate before trusting it.

    Returns:
        ``(observations, features)`` where ``observations`` holds only the rows of
        tracks judged moving, and ``features`` describes every track with its verdict,
        so rejections stay auditable.
    """
    if tracks.empty:
        return tracks.copy(), classify(track_features(tracks), criteria)

    if correct_drift:
        prepared = remove_gps_drift(tracks)
        smoothed = smooth_positions(
            prepared, east_col="east_corrected_m", north_col="north_corrected_m"
        )
    else:
        smoothed = smooth_positions(tracks)
    features = classify(
        track_features(smoothed, "east_smooth_m", "north_smooth_m"), criteria
    )

    keep = set(features.loc[features["is_moving"], "track_id"])
    return smoothed[smoothed["track_id"].isin(keep)].reset_index(drop=True), features


def to_geojson(
    tracks: pd.DataFrame, features: pd.DataFrame, out_path: str | Path
) -> Path:
    """
    Write moving-car paths as a GeoJSON ``FeatureCollection`` of LineStrings.

    One feature per track, carrying its measured properties. GeoJSON rather than CSV
    because it opens directly in any GIS tool or on GitHub.
    """
    lookup = features.set_index("track_id")
    collection = {"type": "FeatureCollection", "features": []}

    for track_id, group in tracks.sort_values("t_sec").groupby("track_id", sort=True):
        row = lookup.loc[track_id] if track_id in lookup.index else None
        collection["features"].append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [round(lon, 7), round(lat, 7)]
                        for lon, lat in zip(group["lon"], group["lat"], strict=True)
                    ],
                },
                "properties": {
                    "track_id": int(track_id),
                    "observations": int(len(group)),
                    "duration_s": round(float(row["duration_s"]), 2) if row is not None else None,
                    "displacement_m": round(float(row["displacement_m"]), 1)
                    if row is not None
                    else None,
                    "median_speed_mps": round(float(row["median_speed_mps"]), 2)
                    if row is not None
                    else None,
                    "median_speed_kmh": round(float(row["median_speed_mps"]) * 3.6, 1)
                    if row is not None
                    else None,
                },
            }
        )

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(collection, indent=1), encoding="utf-8")
    return destination


def report(features: pd.DataFrame) -> str:
    """
    Human-readable summary of what was kept and what was rejected, and why.
    """
    if features.empty:
        return "no tracks"

    counts = features["verdict"].value_counts()
    lines = [
        f"tracks            {len(features)}",
        f"moving            {int(counts.get('moving', 0))}",
    ]
    rejections = (
        "stationary", "wandering", "erratic", "too_slow", "too_short", "implausible_speed",
    )
    for verdict in rejections:
        if verdict in counts:
            lines.append(f"rejected: {verdict:<8} {int(counts[verdict])}")

    moving = features[features["is_moving"]]
    if not moving.empty:
        lines += [
            "",
            f"displacement      {moving['displacement_m'].min():.0f}"
            f" .. {moving['displacement_m'].max():.0f} m",
            f"median speed      {moving['median_speed_mps'].median() * 3.6:.1f} km/h",
            f"duration          {moving['duration_s'].min():.1f}"
            f" .. {moving['duration_s'].max():.1f} s",
        ]
    return "\n".join(lines)
