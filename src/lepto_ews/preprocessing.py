from __future__ import annotations

import pandas as pd


def to_week_start(dts: pd.Series, week_starts_on: str = "MON") -> pd.Series:
    """Normalize timestamps to week start (Mon by default)."""
    dts = pd.to_datetime(dts)
    if week_starts_on.upper() != "MON":
        raise ValueError("Only MON week start is supported for now")
    return (dts.dt.to_period("W-MON").dt.start_time).dt.normalize()


def aggregate_weekly_cases(cases_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    df = cases_df.copy()
    df["week_start"] = to_week_start(df["date"])
    out = (
        df.groupby([id_col, "week_start"], as_index=False)["cases"]
        .sum()
        .sort_values([id_col, "week_start"])
    )
    return out


def aggregate_weekly_climate(climate_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    df = climate_df.copy()
    df["week_start"] = to_week_start(df["date"])

    agg = {"rain_mm": "sum"}
    if "tmean_c" in df.columns:
        agg["tmean_c"] = "mean"
    if "rh_pct" in df.columns:
        agg["rh_pct"] = "mean"

    out = df.groupby([id_col, "week_start"], as_index=False).agg(agg)
    out = out.sort_values([id_col, "week_start"])
    return out


def forward_fill_static(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Used when static columns get merged into weekly rows (keeps values)."""
    return df.groupby(id_col, as_index=False).ffill()
