"""
Map and overlay rendering for flight paths and car tracks.
"""

from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import pandas as pd
from pyproj import Geod

# WGS84 ellipsoid, used for true ground distances instead of degree arithmetic.
_GEOD = Geod(ellps="WGS84")

# Tile layers. Esri's imagery is used because the assignment needs vehicles and
# roads to be visible; the default OSM rendering hides both under labels.
_SATELLITE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/"
    "tile/{z}/{y}/{x}"
)
_SATELLITE_ATTR = "Esri World Imagery"


def path_length_m(lat: np.ndarray, lon: np.ndarray) -> float:
    """
    Total ground distance along a lat/lon polyline, in metres.
    """
    if len(lat) < 2:
        return 0.0
    _, _, seg = _GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    return float(np.nansum(seg))


def base_map(lat: np.ndarray, lon: np.ndarray, zoom: int | None = None) -> folium.Map:
    """
    An empty map centred on the given coordinates, with switchable base layers.
    """
    centre = [float(np.mean(lat)), float(np.mean(lon))]
    fmap = folium.Map(location=centre, zoom_start=zoom or 16, tiles=None, control_scale=True)
    folium.TileLayer(_SATELLITE, attr=_SATELLITE_ATTR, name="Satellite").add_to(fmap)
    folium.TileLayer("OpenStreetMap", name="Street").add_to(fmap)
    return fmap


def add_flight_path(fmap: folium.Map, telemetry: pd.DataFrame, step: int = 10) -> None:
    """
    Draw the drone track with start/end markers and periodic altitude tooltips.

    Args:
        fmap: map to draw on.
        telemetry: table from :mod:`car_tracker.telemetry`.
        step: keep every ``step``-th point. 4979 vertices bloat the HTML and render
            identically to a decimated line at this scale.
    """
    thin = telemetry.iloc[::step]
    points = list(zip(thin["lat"], thin["lon"], strict=True))

    group = folium.FeatureGroup(name="Drone flight path", show=True)
    folium.PolyLine(points, color="#ff3b30", weight=3, opacity=0.9).add_to(group)

    # Sparse altitude probes: enough to see the descent, few enough to stay clickable.
    for _, row in telemetry.iloc[:: max(1, len(telemetry) // 12)].iterrows():
        folium.CircleMarker(
            [row["lat"], row["lon"]],
            radius=3,
            color="#ff3b30",
            fill=True,
            fill_opacity=1.0,
            tooltip=(
                f"frame {int(row['frame'])} · t={row['t_sec']:.1f}s · "
                f"alt {row['rel_alt']:.1f}m · yaw {row['yaw']:.1f}° · "
                f"gsd {row['gsd'] * 100:.1f}cm/px"
            ),
        ).add_to(group)

    first, last = telemetry.iloc[0], telemetry.iloc[-1]
    folium.Marker(
        [first["lat"], first["lon"]],
        tooltip=f"Start · frame {int(first['frame'])} · alt {first['rel_alt']:.1f}m",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(group)
    folium.Marker(
        [last["lat"], last["lon"]],
        tooltip=f"End · frame {int(last['frame'])} · alt {last['rel_alt']:.1f}m",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(group)
    group.add_to(fmap)


# Distinct hues for track polylines, cycled. Chosen to stay legible against both the
# satellite imagery and the street basemap.
TRACK_COLOURS = (
    "#00e5ff", "#ffea00", "#76ff03", "#ff4081",
    "#b388ff", "#ff9100", "#18ffff", "#f4ff81",
)


def add_car_tracks(
    fmap: folium.Map,
    tracks: pd.DataFrame,
    features: pd.DataFrame | None = None,
    name: str = "Moving cars",
    show: bool = True,
) -> None:
    """
    Draw one polyline per car track, with start and end markers.

    Args:
        fmap: map to draw on.
        tracks: observation rows with ``track_id``, ``lat``, ``lon``, ``t_sec``.
        features: optional per-track summary from :mod:`car_tracker.postprocess`,
            used to label each path with its speed and distance.
        name: layer name in the control.
        show: whether the layer starts visible.
    """
    group = folium.FeatureGroup(name=name, show=show)
    summary = features.set_index("track_id") if features is not None else None

    for index, (track_id, rows) in enumerate(tracks.sort_values("t_sec").groupby("track_id")):
        points = list(zip(rows["lat"], rows["lon"], strict=True))
        if len(points) < 2:
            continue
        colour = TRACK_COLOURS[index % len(TRACK_COLOURS)]

        label = f"car {int(track_id)}"
        if summary is not None and track_id in summary.index:
            row = summary.loc[track_id]
            label = (
                f"car {int(track_id)} · {row['median_speed_mps'] * 3.6:.0f} km/h · "
                f"{row['displacement_m']:.0f} m · {row['duration_s']:.1f} s"
            )

        folium.PolyLine(
            points, color=colour, weight=4, opacity=0.95, tooltip=label
        ).add_to(group)
        # Direction of travel is otherwise ambiguous on a bare line.
        folium.CircleMarker(
            points[0], radius=4, color=colour, fill=True, fill_opacity=1.0,
            tooltip=f"{label} — start",
        ).add_to(group)
        folium.RegularPolygonMarker(
            points[-1], number_of_sides=3, radius=7, rotation=0,
            color=colour, fill_color=colour, fill_opacity=1.0,
            tooltip=f"{label} — end",
        ).add_to(group)

    group.add_to(fmap)


def add_rejected_tracks(
    fmap: folium.Map, tracks: pd.DataFrame, features: pd.DataFrame, name: str = "Rejected (parked)"
) -> None:
    """
    Draw tracks that failed the moving test, hidden by default.

    Kept on the map so a rejection can be inspected rather than taken on trust: if a
    genuine car was filtered out, this layer is where it shows up.
    """
    rejected_ids = set(features.loc[~features["is_moving"], "track_id"])
    if not rejected_ids:
        return

    group = folium.FeatureGroup(name=name, show=False)
    summary = features.set_index("track_id")

    for track_id, rows in tracks[tracks["track_id"].isin(rejected_ids)].groupby("track_id"):
        points = list(zip(rows["lat"], rows["lon"], strict=True))
        if len(points) < 2:
            continue
        verdict = summary.loc[track_id, "verdict"]
        folium.PolyLine(
            points, color="#9e9e9e", weight=2, opacity=0.7, dash_array="4",
            tooltip=f"car {int(track_id)} — rejected: {verdict}",
        ).add_to(group)

    group.add_to(fmap)


def render_tracks(
    tracks: pd.DataFrame,
    features: pd.DataFrame,
    out_path: str | Path,
    telemetry: pd.DataFrame | None = None,
    all_tracks: pd.DataFrame | None = None,
) -> Path:
    """
    Write the deliverable map: moving car paths, optionally over the flight path.

    Args:
        tracks: observations of moving cars only.
        features: per-track summary with verdicts.
        out_path: destination HTML file.
        telemetry: if given, the drone's flight path is drawn as context.
        all_tracks: if given, rejected tracks are added as a hidden layer.

    Returns:
        The path written.
    """
    if tracks.empty:
        if telemetry is None:
            raise ValueError("nothing to draw: no tracks and no telemetry")
        lat = telemetry["lat"].to_numpy()
        lon = telemetry["lon"].to_numpy()
    else:
        lat = tracks["lat"].to_numpy()
        lon = tracks["lon"].to_numpy()

    fmap = base_map(lat, lon, zoom=17)
    if telemetry is not None:
        add_flight_path(fmap, telemetry, step=10)
    if all_tracks is not None and not all_tracks.empty:
        add_rejected_tracks(fmap, all_tracks, features)
    if not tracks.empty:
        add_car_tracks(fmap, tracks, features)

    pad = 0.0004  # keep short paths from filling the whole viewport
    fmap.fit_bounds([[lat.min() - pad, lon.min() - pad], [lat.max() + pad, lon.max() + pad]])
    folium.LayerControl(collapsed=False).add_to(fmap)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out))
    return out


def render_flight_path(
    telemetry: pd.DataFrame, out_path: str | Path, step: int = 10
) -> Path:
    """
    Write a standalone HTML map of the drone's flight path.

    Returns:
        The path written.
    """
    lat = telemetry["lat"].to_numpy()
    lon = telemetry["lon"].to_numpy()

    fmap = base_map(lat, lon)
    add_flight_path(fmap, telemetry, step=step)
    fmap.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    folium.LayerControl(collapsed=False).add_to(fmap)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out))
    return out
