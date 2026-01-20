from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from lepto_ews.pipeline import run_end_to_end
from lepto_ews.config import AppConfig


def main() -> None:
    root = ROOT
    data_dir = root / "data" / "sample"
    out_dir = root / "outputs" / "smoke_test"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- boundaries (4 synthetic wards)
    wards = []
    for i in range(2):
        for j in range(2):
            ward_id = f"W{i}{j}"
            geom = box(102.0 + i * 0.1, -18.0 + j * 0.1, 102.0 + (i + 1) * 0.1, -18.0 + (j + 1) * 0.1)
            wards.append({"ward_id": ward_id, "geometry": geom})
    gdf = gpd.GeoDataFrame(wards, crs=4326)
    boundaries_path = data_dir / "wards.gpkg"
    gdf.to_file(boundaries_path, driver="GPKG")

    # --- weekly timeline
    weeks = pd.date_range("2023-01-02", periods=40, freq="W-MON")
    ward_ids = gdf["ward_id"].tolist()

    # --- cases (synthetic outbreaks correlated with rainfall + flood)
    rows = []
    rng = np.random.default_rng(42)
    for wid in ward_ids:
        base = rng.poisson(1.0, size=len(weeks))
        rain = rng.gamma(shape=2.0, scale=10.0, size=len(weeks))
        flood = (rain > np.quantile(rain, 0.8)).astype(int)
        spikes = flood * rng.poisson(4.0, size=len(weeks))
        cases = base + spikes
        for w, c in zip(weeks, cases):
            rows.append({"ward_id": wid, "date": w, "cases": int(c)})
    pd.DataFrame(rows).to_csv(data_dir / "cases.csv", index=False)

    # --- climate
    rows = []
    for wid in ward_ids:
        rain = rng.gamma(shape=2.0, scale=10.0, size=len(weeks))
        tmean = 27 + rng.normal(0, 1.0, size=len(weeks))
        rh = 75 + rng.normal(0, 5.0, size=len(weeks))
        for w, r, t, h in zip(weeks, rain, tmean, rh):
            rows.append({"ward_id": wid, "date": w, "rain_mm": float(r), "tmean_c": float(t), "rh_pct": float(h)})
    pd.DataFrame(rows).to_csv(data_dir / "climate.csv", index=False)

    # --- sanitation (static)
    san = []
    for wid in ward_ids:
        san.append(
            {
                "ward_id": wid,
                "open_drain_pct": float(rng.uniform(0, 60)),
                "no_toilet_pct": float(rng.uniform(0, 60)),
                "unsafe_water_pct": float(rng.uniform(0, 60)),
                "pop_density": float(rng.uniform(500, 5000)),
            }
        )
    pd.DataFrame(san).to_csv(data_dir / "sanitation.csv", index=False)

    # --- flood weekly (simple proxy)
    flood_rows = []
    climate = pd.read_csv(data_dir / "climate.csv")
    climate["date"] = pd.to_datetime(climate["date"])
    for wid in ward_ids:
        sub = climate[climate["ward_id"] == wid].sort_values("date")
        for w, r in zip(sub["date"], sub["rain_mm"]):
            presence = 1 if r > sub["rain_mm"].quantile(0.8) else 0
            flood_rows.append(
                {
                    "ward_id": wid,
                    "week_start": w,
                    "flood_presence": presence,
                    "flooded_area_pct": float(presence * rng.uniform(10, 80)),
                    "flood_duration_days": float(presence * rng.uniform(1, 7)),
                    "flood_frequency_4w": np.nan,
                }
            )
    pd.DataFrame(flood_rows).to_csv(data_dir / "flood_weekly.csv", index=False)

    cfg = AppConfig(
        boundaries_path=boundaries_path,
        boundaries_id_col="ward_id",
        cases_csv=data_dir / "cases.csv",
        climate_csv=data_dir / "climate.csv",
        sanitation_csv=data_dir / "sanitation.csv",
        flood_weekly_csv=data_dir / "flood_weekly.csv",
        prediction_horizon_weeks=1,
        lags_weeks=[1, 2, 3],
        output_dir=out_dir,
    )

    run_end_to_end(cfg)
    print(f"Smoke test complete. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
