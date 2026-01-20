from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import folium
import geopandas as gpd
import pandas as pd
from branca.colormap import linear
from branca.element import MacroElement, Template
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

    # Ensure a single value per (area, week) for clean time animation.
    # Some pipelines can yield duplicates (e.g., multiple intermediate rows).
    df["risk_prob"] = pd.to_numeric(df.get("risk_prob"), errors="coerce").clip(0.0, 1.0)
    if "risk_category" not in df.columns:
        df["risk_category"] = "unknown"

    def _mode_or_unknown(s: pd.Series) -> str:
        s2 = s.dropna()
        if s2.empty:
            return "unknown"
        m = s2.mode()
        return str(m.iloc[0]) if not m.empty else str(s2.iloc[0])

    df = (
        df.groupby([id_col, "week_start"], as_index=False)
        .agg(risk_prob=("risk_prob", "mean"), risk_category=("risk_category", _mode_or_unknown))
        .copy()
    )

    # Cap history to keep the output HTML manageable.
    unique_weeks = sorted(df["week_start"].dropna().unique())
    if len(unique_weeks) > 52:
        keep_weeks = set(unique_weeks[-52:])
        df = df[df["week_start"].isin(keep_weeks)].copy()

    # Cap history to keep the output HTML manageable.
    unique_weeks = sorted(df["week_start"].dropna().unique())
    if len(unique_weeks) > 52:
        keep_weeks = set(unique_weeks[-52:])
        df = df[df["week_start"].isin(keep_weeks)].copy()

    # Build a full grid so every division appears at every time step
    # (missing predictions show as 0 / unknown rather than disappearing).
    boundary_ids = pd.Series(boundaries[id_col].unique(), dtype=object)
    all_weeks = pd.Series(sorted(df["week_start"].dropna().unique()))
    full_index = pd.MultiIndex.from_product([boundary_ids, all_weeks], names=[id_col, "week_start"])
    df = df.set_index([id_col, "week_start"]).reindex(full_index).reset_index()

    prob_for_scale = pd.to_numeric(df["risk_prob"], errors="coerce")
    df["risk_prob"] = prob_for_scale.fillna(0.0).clip(0.0, 1.0)
    df["risk_category"] = df["risk_category"].fillna("unknown")

    df["week_start"] = pd.to_datetime(df["week_start"]).dt.strftime("%Y-%m-%d")
    gdf = _shift_antimeridian_for_leaflet(_simplify_for_web(boundaries)).merge(df, on=id_col, how="left")

    center = _dateline_safe_center(gdf)
    m = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron", control_scale=True, world_copy_jump=True)

    # Always-on outlines so users can hover/click any division even when the
    # choropleth colors are very light.
    base_outline = _shift_antimeridian_for_leaflet(_simplify_for_web(boundaries))
    outline_layer = folium.GeoJson(
        base_outline[[id_col, "geometry"]].to_json(),
        name="Division outlines",
        style_function=lambda _feat: {"color": "#444", "weight": 2, "fillOpacity": 0.0},
    )
    outline_layer.add_to(m)

    # Use robust (quantile) scaling so the map is readable even when
    # probabilities vary a lot across the full time range.
    prob_series = pd.to_numeric(df.get("risk_prob", pd.Series(dtype=float)), errors="coerce")
    if prob_series.notna().any():
        q_low = float(prob_series.quantile(0.02))
        q_high = float(prob_series.quantile(0.98))
        vmin = float(max(0.0, q_low))
        vmax = float(min(1.0, q_high))
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
        # branca returns #RRGGBBAA; strip alpha for broader Leaflet/CSS compatibility
        if isinstance(fill, str) and fill.startswith("#") and len(fill) == 9:
            fill = fill[:7]

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
                "color": "#111",
                "weight": 2,
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

    # Build a lookup so hovering outlines can show the CURRENT slider date's values.
    # Structure: riskLookup["YYYY-MM-DD"][division_id] = [risk_prob, risk_category]
    risk_lookup: dict[str, dict[str, list[object]]] = {}
    for wk, grp in df.groupby("week_start"):
        wk_key = str(wk)
        wk_map: dict[str, list[object]] = {}
        for row in grp.itertuples(index=False):
            div = str(getattr(row, id_col))
            prob = float(getattr(row, "risk_prob"))
            cat = str(getattr(row, "risk_category"))
            wk_map[div] = [prob, cat]
        risk_lookup[wk_key] = wk_map

    map_name = m.get_name()
    outline_name = outline_layer.get_name()
    lookup_json = json.dumps(risk_lookup, separators=(",", ":"))

    macro = MacroElement()
    macro._template = Template(
        f"""
{{% macro script(this, kwargs) %}}
(function() {{
    var map = {map_name};
    var outlines = {outline_name};
    var riskLookup = {lookup_json};

    function toDateKey(t) {{
        if (t === null || t === undefined) return null;
        var d = new Date(t);
        if (isNaN(d)) return null;
        return d.toISOString().slice(0, 10);
    }}

    function getCurrentDateKey() {{
        try {{
            if (map.timeDimension && map.timeDimension.getCurrentTime) {{
                return toDateKey(map.timeDimension.getCurrentTime());
            }}
        }} catch (e) {{}}
        return null;
    }}

    function fmtProb(p) {{
        if (p === null || p === undefined) return 'NA';
        var x = Number(p);
        if (!isFinite(x)) return 'NA';
        return x.toFixed(3);
    }}

    function makeHtml(divId) {{
        var dateKey = getCurrentDateKey();
        var info = (dateKey && riskLookup[dateKey]) ? riskLookup[dateKey][divId] : null;
        var prob = info ? info[0] : null;
        var cat = info ? info[1] : 'unknown';
        return (
            '<b>{id_col}:</b> ' + divId +
            '<br><b>week_start:</b> ' + (dateKey || 'unknown') +
            '<br><b>risk_prob:</b> ' + fmtProb(prob) +
            '<br><b>risk_category:</b> ' + cat
        );
    }}

    // Bind dynamic tooltips/popups to outline polygons.
    outlines.eachLayer(function(layer) {{
        layer.on('mouseover', function(e) {{
            var props = (layer.feature && layer.feature.properties) ? layer.feature.properties : {{}};
            var divId = props['{id_col}'];
            var html = makeHtml(divId);
            layer.bindTooltip(html, {{sticky: true, direction: 'auto', opacity: 0.95}});
            layer.openTooltip(e.latlng);
        }});
        layer.on('mouseout', function() {{
            try {{ layer.closeTooltip(); }} catch (e) {{}}
        }});
        layer.on('click', function(e) {{
            var props = (layer.feature && layer.feature.properties) ? layer.feature.properties : {{}};
            var divId = props['{id_col}'];
            var html = makeHtml(divId);
            layer.bindPopup(html);
            layer.openPopup(e.latlng);
        }});
    }});
}})();
{{% endmacro %}}
"""
    )
    m.get_root().add_child(macro)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_html))
