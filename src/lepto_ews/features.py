from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    id_col: str = "ward_id"


def add_lags(df: pd.DataFrame, id_col: str, time_col: str, cols: list[str], lags: list[int]) -> pd.DataFrame:
    out = df.sort_values([id_col, time_col]).copy()
    for col in cols:
        if col not in out.columns:
            continue
        for lag in lags:
            out[f"{col}_lag{lag}"] = out.groupby(id_col)[col].shift(lag)
    return out


def compute_rain_anomaly(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    *,
    time_aware: bool = True,
    window_weeks: int = 52,
    min_periods: int = 12,
) -> pd.DataFrame:
    out = df.copy()
    if "rain_mm" not in out.columns:
        return out

    if not time_aware:
        mean_by_ward = out.groupby(id_col)["rain_mm"].transform("mean")
        std_by_ward = out.groupby(id_col)["rain_mm"].transform("std").replace(0, np.nan)
        out["rain_anom_z"] = (out["rain_mm"] - mean_by_ward) / std_by_ward
        out["rain_anom_z"] = out["rain_anom_z"].fillna(0.0)
        return out

    out = out.sort_values([id_col, time_col]).copy()

    def _calc_series(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        baseline = s.shift(1)
        roll_mean = baseline.rolling(window_weeks, min_periods=min_periods).mean()
        roll_std = baseline.rolling(window_weeks, min_periods=min_periods).std().replace(0, np.nan)

        return (s - roll_mean) / roll_std

    out["rain_anom_z"] = (
        out.groupby(id_col, group_keys=False)["rain_mm"].apply(_calc_series)
    )
    out["rain_anom_z"] = out["rain_anom_z"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def compute_sanitation_index(san_df: pd.DataFrame) -> pd.DataFrame:
    df = san_df.copy()
    components = ["open_drain_pct", "no_toilet_pct", "unsafe_water_pct"]
    for c in components:
        if c not in df.columns:
            raise ValueError(f"Sanitation component missing: {c}")

    raw = df[components].sum(axis=1)
    minv, maxv = float(raw.min()), float(raw.max())
    if maxv == minv:
        df["sanitation_index"] = 0.0
    else:
        df["sanitation_index"] = (raw - minv) / (maxv - minv)

    return df


def label_outbreaks(
    cases_weekly: pd.DataFrame,
    id_col: str,
    time_col: str,
    method: str,
    quantile: float,
    k_std: float,
    *,
    time_aware: bool = True,
    window_weeks: int = 52,
    min_periods: int = 12,
) -> pd.DataFrame:
    """Create a binary label from observed cases.

    Note: CDC-style thresholding is typically applied to *cases*.
    Here we label for training using a simple, configurable ward-level rule.
    """
    df = cases_weekly.copy()

    df = df.sort_values([id_col, time_col]).copy()

    if not time_aware:
        if method == "quantile":
            thr = df.groupby(id_col)["cases"].transform(lambda s: s.astype(float).quantile(quantile))
        elif method == "mean_plus_kstd":
            mean = df.groupby(id_col)["cases"].transform("mean")
            std = df.groupby(id_col)["cases"].transform("std").fillna(0.0)
            thr = mean + k_std * std
        else:
            raise ValueError(f"Unknown labeling method: {method}")
        df["label_high_risk"] = (df["cases"] > thr).astype(int)
        return df

    def _time_aware_thr_series(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        baseline = s.shift(1)

        if method == "quantile":
            thr = baseline.rolling(window_weeks, min_periods=min_periods).quantile(quantile)
        elif method == "mean_plus_kstd":
            mean = baseline.rolling(window_weeks, min_periods=min_periods).mean()
            std = baseline.rolling(window_weeks, min_periods=min_periods).std().fillna(0.0)
            thr = mean + k_std * std
        else:
            raise ValueError(f"Unknown labeling method: {method}")

        return thr

    thr = df.groupby(id_col, group_keys=False)["cases"].apply(_time_aware_thr_series)

    df["label_high_risk"] = (df["cases"] > thr).astype(int)
    return df


def apply_risk_thresholds(pred_df: pd.DataFrame, id_col: str, prob_col: str, low: float, high: float, consecutive_weeks: int) -> pd.DataFrame:
    df = pred_df.sort_values([id_col, "week_start"]).copy()

    def category(p: float) -> str:
        if p < low:
            return "low"
        if p > high:
            return "high"
        return "medium"

    df["risk_category_raw"] = df[prob_col].apply(category)

    # Consecutive high-risk rule
    is_high = (df["risk_category_raw"] == "high").astype(int)
    streak = is_high.groupby(df[id_col]).transform(lambda s: s.rolling(consecutive_weeks).sum())
    df["risk_category"] = df["risk_category_raw"]
    df.loc[streak < consecutive_weeks, "risk_category"] = df.loc[streak < consecutive_weeks, "risk_category"].replace({"high": "medium"})

    return df
