from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkb


def _force_2d_geometry(geom):
    if geom is None:
        return None
    # Strip Z/M dimensions in a Shapely-version-tolerant way.
    return wkb.loads(wkb.dumps(geom, output_dimension=2))


def read_boundaries(path: Path, id_col: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if id_col not in gdf.columns:
        raise ValueError(f"Boundary id column '{id_col}' not found. Available: {list(gdf.columns)}")
    gdf = gdf[[id_col, "geometry"]].copy()
    gdf[id_col] = gdf[id_col].astype(str)
    gdf["geometry"] = gdf["geometry"].apply(_force_2d_geometry)
    gdf = gdf.to_crs(4326)
    return gdf


def read_cases_csv(path: Path, id_col: str = "ward_id") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {id_col, "date", "cases"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cases CSV missing columns: {sorted(missing)}")
    df = df[[id_col, "date", "cases"]].copy()
    df[id_col] = df[id_col].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df["cases"] = pd.to_numeric(df["cases"], errors="coerce").fillna(0.0)
    return df


def read_climate_csv(path: Path, id_col: str = "ward_id") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {id_col, "date", "rain_mm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Climate CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    for col in ["rain_mm", "tmean_c", "rh_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_sanitation_csv(path: Path, id_col: str = "ward_id") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {id_col, "open_drain_pct", "no_toilet_pct", "unsafe_water_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sanitation CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    for col in ["open_drain_pct", "no_toilet_pct", "unsafe_water_pct", "pop_density"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_flood_weekly_csv(path: Path, id_col: str = "ward_id") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {id_col, "week_start"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Flood weekly CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    df["week_start"] = pd.to_datetime(df["week_start"]).dt.normalize()
    for col in ["flooded_area_pct", "flood_presence", "flood_duration_days", "flood_frequency_4w"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
