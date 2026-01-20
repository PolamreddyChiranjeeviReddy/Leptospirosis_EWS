"""Download CHIRPS rainfall and aggregate to Fiji divisions.

This script is designed to be beginner-friendly:
- It auto-detects the date range from data/fiji/cases.csv
- Downloads CHIRPS yearly NetCDF files (one year at a time)
- Computes area-mean daily rainfall (mm/day) per division polygon
- Writes data/fiji/climate.csv with columns: division_id,date,rain_mm

Notes
-----
- Download source: CHIRPS v2 global daily, 0.05 degree, NetCDF (p05)
  https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/
- The CHIRPS NetCDF files are ~1.1GB/year. This script processes each year
    sequentially. By default it DOES NOT delete any downloaded files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely import wkb
from shapely.validation import make_valid

try:
    import geopandas as gpd
    from rasterio.transform import from_origin
    from rasterio.features import rasterize
    import netCDF4
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing GIS dependencies. Ensure you installed requirements.txt in your venv.\n"
        f"Original error: {type(exc).__name__}: {exc}"
    )


CHIRPS_BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    def reporthook(blocknum: int, blocksize: int, totalsize: int) -> None:
        if totalsize <= 0:
            return
        downloaded = blocknum * blocksize
        pct = min(100.0, downloaded * 100.0 / totalsize)
        sys.stdout.write(f"\rDownloading {dest.name}: {pct:5.1f}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=reporthook)
    sys.stdout.write("\n")
    sys.stdout.flush()
    tmp.replace(dest)


def _lon_to_dataset_range(lon: float, lon_min: float, lon_max: float) -> float:
    """Convert lon to match the dataset's longitude convention."""
    # CHIRPS may use 0..360 or -180..180. If dataset is 0..360, convert negatives.
    if lon_min >= 0 and lon_max > 180:
        return lon if lon >= 0 else lon + 360.0
    # If dataset is -180..180, convert values > 180.
    if lon_min < 0 and lon_max <= 180:
        return lon if lon <= 180 else lon - 360.0
    return lon


def _subset_indices(coords: np.ndarray, vmin: float, vmax: float) -> tuple[int, int]:
    """Return [start, stop) indices covering vmin..vmax in a 1D coord array."""
    if coords[0] < coords[-1]:
        start = int(np.searchsorted(coords, vmin, side="left"))
        stop = int(np.searchsorted(coords, vmax, side="right"))
    else:
        # descending
        rev = coords[::-1]
        start_r = int(np.searchsorted(rev, vmin, side="left"))
        stop_r = int(np.searchsorted(rev, vmax, side="right"))
        n = len(coords)
        start = n - stop_r
        stop = n - start_r
    start = max(0, min(start, len(coords) - 1))
    stop = max(start + 1, min(stop, len(coords)))
    return start, stop


def _date_range_from_cases(cases_csv: Path, id_col: str) -> tuple[dt.date, dt.date]:
    df = pd.read_csv(cases_csv)
    if "date" not in df.columns:
        raise ValueError(f"cases CSV missing 'date': {cases_csv}")
    if id_col not in df.columns:
        raise ValueError(f"cases CSV missing '{id_col}': {cases_csv}")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df["date"].min(), df["date"].max()


def _load_boundaries(boundaries_path: Path, id_col: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(boundaries_path)
    if id_col not in gdf.columns:
        raise ValueError(f"Boundary id column '{id_col}' not found in {boundaries_path}. Columns: {list(gdf.columns)}")
    gdf = gdf[[id_col, "geometry"]].copy()
    gdf[id_col] = gdf[id_col].astype(str)
    # Ensure lat/lon
    try:
        gdf = gdf.to_crs(4326)
    except Exception:
        # If CRS missing, assume it's already lat/lon.
        pass

    # Force 2D and repair invalid geometries (common in admin boundary datasets).
    # This prevents TopologyException during spatial ops/zonal stats.
    def _fix(geom):
        if geom is None:
            return None
        try:
            # Strip Z/M dimensions if present.
            geom = wkb.loads(wkb.dumps(geom, output_dimension=2))
            if not geom.is_valid:
                return make_valid(geom)
            return geom
        except Exception:
            # Fallback heuristic
            try:
                return geom.buffer(0)
            except Exception:
                return geom

    gdf["geometry"] = gdf["geometry"].apply(_fix)
    gdf = gdf[~gdf["geometry"].isna()].copy()
    return gdf


def _iter_dates_for_year(year: int, band_count: int) -> list[dt.date]:
    start = dt.date(year, 1, 1)
    dates = [start + dt.timedelta(days=i) for i in range(band_count)]
    return dates


def _chirps_daily_means_for_date(
    nc_path: Path,
    boundaries: gpd.GeoDataFrame,
    id_col: str,
    start_date: dt.date,
    end_date: dt.date,
) -> list[dict]:
    """Compute daily mean rainfall per polygon for the date window that overlaps this file."""
    ds = netCDF4.Dataset(str(nc_path), mode="r")
    try:
        if "precip" not in ds.variables:
            raise RuntimeError(f"CHIRPS file missing 'precip' variable: {nc_path}")

        precip = ds.variables["precip"]

        # Coordinates
        # Common CHIRPS names are latitude/longitude, but handle a few variants.
        lat_name = "latitude" if "latitude" in ds.variables else ("lat" if "lat" in ds.variables else None)
        lon_name = "longitude" if "longitude" in ds.variables else ("lon" if "lon" in ds.variables else None)
        time_name = "time" if "time" in ds.variables else None
        if lat_name is None or lon_name is None or time_name is None:
            raise RuntimeError(f"Unexpected CHIRPS coordinate variables in {nc_path}. Vars: {list(ds.variables.keys())}")

        lats = np.asarray(ds.variables[lat_name][:], dtype=float)
        lons = np.asarray(ds.variables[lon_name][:], dtype=float)
        time_var = ds.variables[time_name]
        times = netCDF4.num2date(time_var[:], units=time_var.units, only_use_cftime_datetimes=False)
        dates = [t.date() for t in times]

        # Subset to Fiji bounding box (+ padding) to keep processing fast.
        minx, miny, maxx, maxy = boundaries.total_bounds
        pad = 0.5
        lon_min = float(lons.min())
        lon_max = float(lons.max())
        minx2 = _lon_to_dataset_range(minx - pad, lon_min, lon_max)
        maxx2 = _lon_to_dataset_range(maxx + pad, lon_min, lon_max)
        # If conversion caused wrap, just take full range.
        if minx2 > maxx2:
            minx2, maxx2 = min(lon_min, lon_max), max(lon_min, lon_max)

        x0, x1 = _subset_indices(lons, minx2, maxx2)
        y0, y1 = _subset_indices(lats, miny - pad, maxy + pad)

        lons_sub = lons[x0:x1]
        lats_sub = lats[y0:y1]

        # Raster grids assume row 0 is NORTH. If dataset lats ascend (south->north),
        # we flip the extracted arrays so row 0 corresponds to the northernmost latitude.
        flip_y = bool(len(lats_sub) > 1 and lats_sub[0] < lats_sub[-1])
        lats_desc = lats_sub[::-1] if flip_y else lats_sub

        # Affine transform for the subset grid (assumes coords are cell centers).
        dx = float(abs(lons_sub[1] - lons_sub[0])) if len(lons_sub) > 1 else 0.05
        dy = float(abs(lats_desc[1] - lats_desc[0])) if len(lats_desc) > 1 else 0.05
        west = float(min(lons_sub) - dx / 2.0)
        north = float(max(lats_desc) + dy / 2.0)
        transform = from_origin(west, north, dx, dy)

        # Do NOT intersection-clip polygons: it can raise TopologyException on imperfect geometries.
        # We already subset the raster window; we'll rasterize the polygons once.
        b2 = boundaries.copy()

        out: list[dict] = []
        area_ids = [str(x) for x in b2[id_col].tolist()]

        # Rasterize polygons ONCE for this file+subset.
        # label 0 = background; labels 1..N correspond to area_ids order.
        shapes = []
        for idx, geom in enumerate(b2.geometry.tolist(), start=1):
            if geom is None or geom.is_empty:
                continue
            shapes.append((geom, idx))

        label_grid = rasterize(
            shapes,
            out_shape=(len(lats_desc), len(lons_sub)),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="int32",
        )
        labels_flat = label_grid.ravel()

        for ti, d in enumerate(dates):
            if d < start_date or d > end_date:
                continue

            # Load only the Fiji window for this day.
            # precip dims are typically [time, lat, lon]
            try:
                arr = np.asarray(precip[ti, y0:y1, x0:x1], dtype=float)
                if flip_y:
                    arr = arr[::-1, :]
            except Exception as exc:
                print(f"Warning: failed reading CHIRPS slice {nc_path.name} day={d}: {type(exc).__name__}: {exc}")
                continue
            # CHIRPS uses -9999 for missing
            nodata = getattr(precip, "_FillValue", -9999.0)

            arr_flat = arr.ravel()
            valid = (labels_flat > 0) & (arr_flat != float(nodata))
            if not np.any(valid):
                for area_id in area_ids:
                    out.append({id_col: area_id, "date": d.isoformat(), "rain_mm": 0.0})
                continue

            sums = np.bincount(labels_flat[valid], weights=arr_flat[valid], minlength=len(area_ids) + 1)
            counts = np.bincount(labels_flat[valid], minlength=len(area_ids) + 1)
            means = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)

            for idx, area_id in enumerate(area_ids, start=1):
                out.append({id_col: area_id, "date": d.isoformat(), "rain_mm": float(means[idx])})

        return out
    finally:
        ds.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Fiji climate.csv from CHIRPS rainfall")
    parser.add_argument("--cases-csv", type=Path, default=Path("data/fiji/cases.csv"))
    parser.add_argument("--boundaries", type=Path, default=Path("data/fiji/fiji_adm1_divisions.gpkg"))
    parser.add_argument("--id-col", type=str, default="division_id")
    parser.add_argument("--out-csv", type=Path, default=Path("data/fiji/climate.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/fiji/_chirps_cache"))
    parser.add_argument(
        "--years",
        type=int,
        default=1,
        help="Download/process only the last N years within the cases date range (default: 1).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Optional explicit start year (overrides --years).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Optional explicit end year (overrides --years).",
    )
    parser.add_argument(
        "--delete-yearly-files",
        action="store_true",
        help="Delete downloaded yearly NetCDF files after processing (OFF by default).",
    )
    args = parser.parse_args()

    start_date, end_date = _date_range_from_cases(args.cases_csv, args.id_col)
    print(f"Cases date range: {start_date} .. {end_date}")

    boundaries = _load_boundaries(args.boundaries, args.id_col)

    # Choose a limited year range (disk-friendly).
    cases_start_year = start_date.year
    cases_end_year = end_date.year
    if args.start_year is not None or args.end_year is not None:
        start_year = args.start_year if args.start_year is not None else cases_start_year
        end_year = args.end_year if args.end_year is not None else cases_end_year
    else:
        n_years = max(1, int(args.years))
        end_year = cases_end_year
        start_year = max(cases_start_year, cases_end_year - n_years + 1)

    years = list(range(start_year, end_year + 1))
    print(f"CHIRPS years selected: {years[0]}..{years[-1]} ({len(years)} year(s))")
    records: list[dict] = []

    # Reduce noisy GDAL warnings
    os.environ.setdefault("CPL_LOG", "NUL" if os.name == "nt" else "/dev/null")

    for year in years:
        fname = f"chirps-v2.0.{year}.days_p05.nc"
        url = CHIRPS_BASE_URL + fname
        local_nc = args.cache_dir / fname

        print(f"\n=== {year} ===")
        print(f"Source: {url}")
        _download(url, local_nc)

        records.extend(
            _chirps_daily_means_for_date(
                local_nc,
                boundaries=boundaries,
                id_col=args.id_col,
                start_date=start_date,
                end_date=end_date,
            )
        )

        if args.delete_yearly_files:
            try:
                local_nc.unlink()
            except Exception:
                pass

    out_df = pd.DataFrame.from_records(records)
    if out_df.empty:
        raise RuntimeError("No climate rows produced. Check cases date range and boundary IDs.")

    # Ensure stable ordering
    out_df["date"] = pd.to_datetime(out_df["date"]).dt.date.astype(str)
    out_df = out_df.sort_values([args.id_col, "date"]).reset_index(drop=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nWrote: {args.out_csv} (rows={len(out_df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
