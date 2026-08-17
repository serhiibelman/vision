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
