"""Aggregate a WorldPop population density raster to Fiji divisions.

You download ONE GeoTIFF from WorldPop (population density or population count).
Then this script computes a mean density per division polygon and writes it
into the sanitation file schema (required columns present).

Download location (WorldPop): https://www.worldpop.org/
Suggested dataset: population density GeoTIFF for Fiji.

Inputs
------
- --raster: WorldPop GeoTIFF (density preferred)
- --boundaries: division polygons (GPKG/SHP)

Outputs
-------
- data/fiji/sanitation.csv with required columns.

Note
----
If you only find a *population count* raster, we can still compute density by
also computing polygon area, but density rasters are simpler.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import LineString
from shapely.ops import split as shapely_split

try:
    from shapely.validation import make_valid  # type: ignore
except Exception:  # pragma: no cover
    make_valid = None


def _explode_antimeridian(gdf: gpd.GeoDataFrame, id_col: str) -> gpd.GeoDataFrame:
    """Split geometries that cross the antimeridian (180°) into parts.

    Without this, bounds can span almost the whole world, causing rasterstats to
    try to read an enormous window from global rasters.
    """

    cut = LineString([(180.0, -90.0), (180.0, 90.0)])
    rows: list[dict] = []
    for _, r in gdf.iterrows():
        geom = r.geometry
        if geom is None or geom.is_empty:
            continue
        minx, _, maxx, _ = geom.bounds
        if (maxx - minx) > 180.0:
            try:
                parts = shapely_split(geom, cut)
                for part in getattr(parts, "geoms", [parts]):
                    if part is not None and not part.is_empty:
                        rows.append({id_col: r[id_col], "geometry": part})
            except Exception:
                rows.append({id_col: r[id_col], "geometry": geom})
        else:
            rows.append({id_col: r[id_col], "geometry": geom})

    return gpd.GeoDataFrame(rows, crs=gdf.crs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare sanitation.csv with pop_density from WorldPop")
    parser.add_argument("--raster", type=Path, required=True, help="WorldPop GeoTIFF (population density preferred)")
    parser.add_argument("--boundaries", type=Path, default=Path("data/fiji/fiji_adm1_divisions.gpkg"))
    parser.add_argument("--id-col", type=str, default="division_id")
    parser.add_argument("--out", type=Path, default=Path("data/fiji/sanitation.csv"))
    args = parser.parse_args()

    gdf = gpd.read_file(args.boundaries)
    if args.id_col not in gdf.columns:
        raise SystemExit(f"Missing id column '{args.id_col}' in {args.boundaries}. Columns: {list(gdf.columns)}")
    gdf = gdf[[args.id_col, "geometry"]].copy()
    gdf[args.id_col] = gdf[args.id_col].astype(str)

    # Repair invalid polygons (common in admin boundary datasets)
    if not gdf.geometry.is_valid.all():
        if make_valid is not None:
            gdf["geometry"] = gdf.geometry.apply(make_valid)
        else:
            gdf["geometry"] = gdf.geometry.buffer(0)

    # Pre-compute polygon area for density conversion when using "ppp" (people per pixel) rasters.
    # Use a global equal-area CRS so areas are meaningful.
    area_km2_by_id = (
        gdf.to_crs(6933)
        .assign(_area_km2=lambda d: d.geometry.area.astype(float) / 1_000_000.0)
        .groupby(args.id_col, as_index=True)["_area_km2"]
        .sum()
    )

    raster_name = args.raster.name.lower()
    is_ppp = "_ppp_" in raster_name or raster_name.endswith("ppp.tif") or "ppp" in raster_name

    # IMPORTANT: reproject polygons to the raster CRS before zonal stats.
    # Otherwise rasterstats may try to read an astronomically large window.
    with rasterio.open(args.raster) as ds:
        nodata = ds.nodata
        raster_crs = ds.crs

    gdf_zonal = gdf
    if raster_crs is not None and gdf_zonal.crs is not None and gdf_zonal.crs != raster_crs:
        gdf_zonal = gdf_zonal.to_crs(raster_crs)

    # Split antimeridian-crossing polygons so raster windows stay small.
    gdf_zonal = _explode_antimeridian(gdf_zonal, args.id_col)

    # Pass the raster path (not ds.read(1)) so rasterstats can window-read only
    # the pixels intersecting Fiji polygons.
        stats = zonal_stats(
            gdf_zonal,
            str(args.raster),
            stats=["mean", "sum"],
            nodata=nodata,
            all_touched=True,
        )

    stats_df = pd.DataFrame(stats)
    stats_df[args.id_col] = gdf_zonal[args.id_col].astype(str).to_numpy()

    if is_ppp:
        total_pop_by_id = stats_df.groupby(args.id_col, as_index=True)["sum"].sum(min_count=1)
        pop_density_by_id = {}
        for division_id, total_pop in total_pop_by_id.items():
            a = float(area_km2_by_id.get(division_id, 0.0))
            if pd.isna(total_pop) or a <= 0:
                pop_density_by_id[division_id] = None
            else:
                pop_density_by_id[division_id] = float(total_pop) / a
    else:
        # For true density rasters, compute an area-weighted mean across parts.
        parts_area = (
            gdf_zonal.to_crs(6933)
            .assign(_area_km2=lambda d: d.geometry.area.astype(float) / 1_000_000.0)
            .reset_index(drop=True)
        )
        stats_df = stats_df.reset_index(drop=True)
        stats_df["_area_km2"] = parts_area["_area_km2"].to_numpy()

        def _weighted_mean(group: pd.DataFrame) -> float | None:
            g = group.dropna(subset=["mean", "_area_km2"])
            if g.empty:
                return None
            denom = float(g["_area_km2"].sum())
            if denom <= 0:
                return None
            return float((g["mean"] * g["_area_km2"]).sum() / denom)

        mean_by_id = stats_df.groupby(args.id_col, as_index=True).apply(_weighted_mean)
        pop_density_by_id = {k: (None if pd.isna(v) else float(v)) for k, v in mean_by_id.items()}

    pop_density = [pop_density_by_id.get(v) for v in gdf[args.id_col].tolist()]

    out = pd.DataFrame(
        {
            args.id_col: gdf[args.id_col].tolist(),
            "open_drain_pct": [None] * len(gdf),
            "no_toilet_pct": [None] * len(gdf),
            "unsafe_water_pct": [None] * len(gdf),
            "pop_density": pop_density,
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
