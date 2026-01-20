from __future__ import annotations

from pathlib import Path

import numpy as np
import folium
import geopandas as gpd
import pandas as pd
from branca.colormap import linear
from folium.plugins import TimestampedGeoJson
from shapely.ops import transform


def _simplify_for_web(gdf: gpd.GeoDataFrame, tolerance_m: float = 1000.0) -> gpd.GeoDataFrame:
    """Simplify polygons to keep GeoJSON/HTML outputs small.

    Uses a metric CRS (EPSG:3857) for a stable tolerance in meters.
    """
    if gdf.empty:
        return gdf
    g = gdf.copy()
    g = g.to_crs(3857)
    g["geometry"] = g["geometry"].simplify(tolerance_m, preserve_topology=True)
    g = g.to_crs(4326)
    return g


def _dateline_safe_center(gdf: gpd.GeoDataFrame) -> list[float]:
    """Compute a map center that behaves when geometries cross the antimeridian.

    Fiji (and other Pacific geometries) can span both +180 and -180 longitudes.
    A naive bounds-average often yields ~0 longitude (Greenwich), making the map
    look "empty" even though data is present.
    """
    if gdf.empty:
        return [0.0, 0.0]

    g = gdf.to_crs(4326)
    pts = g.geometry.representative_point()
    lons = np.array([p.x for p in pts if p is not None], dtype=float)
    lats = np.array([p.y for p in pts if p is not None], dtype=float)
    if lons.size == 0 or lats.size == 0:
        return [0.0, 0.0]

    # If span suggests dateline crossing, shift negative longitudes into [0, 360)
    # before averaging, then shift back to [-180, 180].
    lon_span = float(np.nanmax(lons) - np.nanmin(lons))
    if lon_span > 180.0:
        lons_shift = np.where(lons < 0.0, lons + 360.0, lons)
        lon_mean = float(np.nanmean(lons_shift))
        if lon_mean > 180.0:
            lon_mean -= 360.0
    else:
        lon_mean = float(np.nanmean(lons))

    lat_mean = float(np.nanmean(lats))
    return [lat_mean, lon_mean]


def _shift_antimeridian_for_leaflet(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Shift geometries that cross the dateline into a consistent 0..360 longitude space.

    Some Pacific polygons can be represented in a way that makes them span almost the
    entire world in Leaflet (because the polygon "connects" across -180/180). Shifting
    negative longitudes by +360 keeps the shape compact around ~180E for web maps.

    Note: This should be applied AFTER any CRS transformations/simplification.
    """
    if gdf.empty:
        return gdf

    g = gdf.to_crs(4326).copy()
    minx, _, maxx, _ = g.total_bounds
    if (maxx - minx) <= 180.0:
        return g

    def _shift(x, y, z=None):
        x = np.asarray(x)
        x2 = np.where(x < 0.0, x + 360.0, x)
        if z is None:
            return (x2, y)
        return (x2, y, z)

    g["geometry"] = g["geometry"].apply(lambda geom: transform(_shift, geom) if geom is not None else None)
    return g


def make_latest_risk_map(boundaries: gpd.GeoDataFrame, risk_df: pd.DataFrame, id_col: str, output_html: Path) -> None:
    latest_week = pd.to_datetime(risk_df["week_start"]).max()
    latest = risk_df[risk_df["week_start"] == latest_week].copy()

    latest = latest[[id_col, "risk_prob", "risk_category"]]
    gdf = boundaries.merge(latest, on=id_col, how="left")
    gdf = _shift_antimeridian_for_leaflet(_simplify_for_web(gdf))

    center = _dateline_safe_center(gdf)
    m = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron", control_scale=True, world_copy_jump=True)

    folium.Choropleth(
        geo_data=gdf.to_json(),
        data=gdf,
        columns=[id_col, "risk_prob"],
        key_on=f"feature.properties.{id_col}",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Predicted risk probability",
    ).add_to(m)

    folium.GeoJson(
        gdf[[id_col, "risk_prob", "risk_category", "geometry"]].to_json(),
        tooltip=folium.GeoJsonTooltip(fields=[id_col, "risk_prob", "risk_category"]),
    ).add_to(m)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_html))


def make_timeslider_map(boundaries: gpd.GeoDataFrame, risk_df: pd.DataFrame, id_col: str, output_html: Path) -> None:
    df = risk_df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"]).dt.normalize()

    # Cap history to keep the output HTML manageable.
    unique_weeks = sorted(df["week_start"].dropna().unique())
    if len(unique_weeks) > 52:
        keep_weeks = set(unique_weeks[-52:])
        df = df[df["week_start"].isin(keep_weeks)].copy()

    df["week_start"] = df["week_start"].dt.strftime("%Y-%m-%d")
    gdf = _shift_antimeridian_for_leaflet(_simplify_for_web(boundaries)).merge(df, on=id_col, how="left")

    center = _dateline_safe_center(gdf)
    m = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron", control_scale=True, world_copy_jump=True)

    prob_series = pd.to_numeric(df.get("risk_prob", pd.Series(dtype=float)), errors="coerce")
    if prob_series.notna().any():
        vmin = float(max(0.0, prob_series.min()))
        vmax = float(min(1.0, prob_series.max()))
        if vmax <= vmin:
            vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, 1.0

    colormap = linear.YlOrRd_09.scale(vmin, vmax)
    colormap.caption = "Predicted risk probability"
    colormap.add_to(m)

    features = []
    gdf2 = gdf.dropna(subset=["week_start"]).copy()
    gdf2["risk_prob"] = pd.to_numeric(gdf2.get("risk_prob"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    for _, row in gdf2.iterrows():
        prob = float(row.get("risk_prob", 0.0))
        cat = row.get("risk_category", "unknown")
        fill = colormap(prob)

        tooltip = (
            f"{id_col}: {row[id_col]}"
            f"<br>week_start: {row['week_start']}"
            f"<br>risk_prob: {prob:.3f}"
            f"<br>risk_category: {cat}"
        )
        props = {
            "time": row["week_start"],
            id_col: row[id_col],
            "risk_prob": prob,
            "risk_category": cat,
            "tooltip": tooltip,
            "popup": tooltip,
            "style": {
                "color": "black",
                "weight": 1,
                "fillColor": fill,
                "fillOpacity": 0.65,
            },
        }
        features.append({"type": "Feature", "geometry": row.geometry.__geo_interface__, "properties": props})

    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period="P7D",
        add_last_point=False,
        auto_play=False,
        loop=False,
        max_speed=5,
        loop_button=True,
        date_options="YYYY-MM-DD",
        time_slider_drag_update=True,
    ).add_to(m)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_html))
