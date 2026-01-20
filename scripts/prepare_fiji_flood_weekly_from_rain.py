"""Create a weekly flood indicator table from rainfall (proxy).

Why this exists
---------------
Open, truly weekly flood extent datasets are harder to obtain than rainfall.
A practical, paper-defendable first step is to create a *proxy* flood signal
from extreme rainfall using CHIRPS.

Input
-----
- data/fiji/climate.csv with columns: division_id,date,rain_mm (daily)

Output
------
- data/fiji/flood_weekly.csv with columns:
  - division_id
  - week_start
  - weekly_rain_mm
  - flood_presence (0/1)  [proxy]
  - flood_frequency_4w    (rolling count of flood weeks)

Logic (simple + transparent)
----------------------------
- Aggregate daily rain -> weekly total per division.
- Compute trailing 52-week 90th percentile per division.
- flood_presence = 1 if weekly_rain_mm >= q90 (and enough history), else 0.
- flood_frequency_4w = rolling sum of flood_presence over last 4 weeks.

This keeps the pipeline fully open-data (CHIRPS + boundaries) and provides a
flood-like covariate until a true inundation dataset is integrated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Fiji flood_weekly.csv from rainfall proxy")
    parser.add_argument("--climate", type=Path, default=Path("data/fiji/climate.csv"))
    parser.add_argument("--id-col", type=str, default="division_id")
    parser.add_argument("--out", type=Path, default=Path("data/fiji/flood_weekly.csv"))
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--history-weeks", type=int, default=52)
    parser.add_argument("--min-periods", type=int, default=26)
    args = parser.parse_args()

    df = pd.read_csv(args.climate)
    required = {args.id_col, "date", "rain_mm"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Climate CSV missing columns: {sorted(missing)}")

    df[args.id_col] = df[args.id_col].astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["rain_mm"] = pd.to_numeric(df["rain_mm"], errors="coerce").fillna(0.0)

    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    wk = (
        df.groupby([args.id_col, "week_start"], as_index=False)
        .agg(weekly_rain_mm=("rain_mm", "sum"))
        .sort_values([args.id_col, "week_start"])
        .reset_index(drop=True)
    )

    def add_flood_signal(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("week_start").copy()
        q = g["weekly_rain_mm"].rolling(args.history_weeks, min_periods=args.min_periods).quantile(args.quantile)
        g["flood_presence"] = (g["weekly_rain_mm"] >= q).astype(int)
        g.loc[q.isna(), "flood_presence"] = 0
        g["flood_frequency_4w"] = g["flood_presence"].rolling(4, min_periods=1).sum().astype(int)
        return g

    wk = wk.groupby(args.id_col, group_keys=False).apply(add_flood_signal)

    out = wk[[args.id_col, "week_start", "weekly_rain_mm", "flood_presence", "flood_frequency_4w"]].copy()
    out["week_start"] = pd.to_datetime(out["week_start"]).dt.normalize()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote: {args.out} (rows={len(out)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
