from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rich.progress import track

from lepto_ews.config import AppConfig
from lepto_ews.explainability.shap_utils import save_shap_artifacts
from lepto_ews.features import (
    add_lags,
    apply_risk_thresholds,
    compute_rain_anomaly,
    compute_sanitation_index,
    label_outbreaks,
)
from lepto_ews.gis.mapping import make_latest_risk_map, make_timeslider_map
from lepto_ews.io import (
    read_boundaries,
    read_cases_csv,
    read_climate_csv,
    read_flood_weekly_csv,
    read_sanitation_csv,
)
from lepto_ews.modeling.metrics import hit_false_alarm_rates, pr_auc, safe_auc
from lepto_ews.modeling.tuning import time_based_train_val_split, tune_xgb_params
from lepto_ews.modeling.validation import rolling_time_splits
from lepto_ews.modeling.calibration import calibrate_prefit
from lepto_ews.modeling.xgb_model import predict_xgb, train_xgb
from lepto_ews.preprocessing import aggregate_weekly_cases, aggregate_weekly_climate
from lepto_ews.run_metadata import finalize_run_metadata, write_run_metadata


def build_ward_week_table(cfg: AppConfig) -> tuple[pd.DataFrame, list[str]]:
    cases = aggregate_weekly_cases(read_cases_csv(cfg.cases_csv, cfg.boundaries_id_col), cfg.boundaries_id_col)
    df = cases.copy()

    if cfg.climate_csv is not None:
        climate = aggregate_weekly_climate(read_climate_csv(cfg.climate_csv, cfg.boundaries_id_col), cfg.boundaries_id_col)
        df = df.merge(climate, on=[cfg.boundaries_id_col, "week_start"], how="left")
        df = compute_rain_anomaly(
            df,
            cfg.boundaries_id_col,
            "week_start",
            time_aware=cfg.features.rain_anomaly.time_aware,
            window_weeks=cfg.features.rain_anomaly.window_weeks,
            min_periods=cfg.features.rain_anomaly.min_periods,
        )

    if cfg.flood_weekly_csv is not None:
        flood = read_flood_weekly_csv(cfg.flood_weekly_csv, cfg.boundaries_id_col)
        df = df.merge(flood, on=[cfg.boundaries_id_col, "week_start"], how="left")

    # Merge static sanitation (optional)
    if cfg.sanitation_csv is not None:
        sanitation = compute_sanitation_index(read_sanitation_csv(cfg.sanitation_csv, cfg.boundaries_id_col))
        df = df.merge(sanitation, on=cfg.boundaries_id_col, how="left")

    # Time features (helpful even when climate is unavailable)
    week_dt = pd.to_datetime(df["week_start"])
    df["month"] = week_dt.dt.month.astype(int)
    df["week_of_year"] = week_dt.dt.isocalendar().week.astype(int)
    df["sin_woy"] = np.sin(2 * np.pi * df["week_of_year"] / 52.0)
    df["cos_woy"] = np.cos(2 * np.pi * df["week_of_year"] / 52.0)

    # Feature lags (only for columns that exist)
    lag_cols = [
        "cases",
        "rain_mm",
        "tmean_c",
        "rh_pct",
        "rain_anom_z",
        "flooded_area_pct",
        "flood_presence",
    ]
    df = add_lags(df, cfg.boundaries_id_col, "week_start", lag_cols, cfg.lags_weeks)

    df = label_outbreaks(
        df,
        id_col=cfg.boundaries_id_col,
        time_col="week_start",
        method=cfg.labeling.method,
        quantile=cfg.labeling.quantile,
        k_std=cfg.labeling.k_std,
        time_aware=cfg.labeling.time_aware,
        window_weeks=cfg.labeling.window_weeks,
        min_periods=cfg.labeling.min_periods,
    )

    # Select feature columns present
    candidate_features = [
        "rain_mm",
        "rain_anom_z",
        "tmean_c",
        "rh_pct",
        "sanitation_index",
        "pop_density",
        "month",
        "week_of_year",
        "sin_woy",
        "cos_woy",
        "flooded_area_pct",
        "flood_presence",
        "flood_duration_days",
        "flood_frequency_4w",
    ]

    for base in ["cases", "rain_mm", "tmean_c", "rh_pct", "rain_anom_z", "flooded_area_pct", "flood_presence"]:
        for lag in cfg.lags_weeks:
            candidate_features.append(f"{base}_lag{lag}")

    feature_columns = [c for c in candidate_features if c in df.columns]

    return df, feature_columns


def run_xgb_rolling(cfg: AppConfig, df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    df = df.sort_values([cfg.boundaries_id_col, "week_start"]).copy()

    # Minimal cleaning: keep rows with labels; XGBoost can handle missing features.
    model_df = df.dropna(subset=["label_high_risk"]).copy()

    weeks = sorted(model_df["week_start"].unique())
    splits = rolling_time_splits(weeks, horizon_weeks=cfg.prediction_horizon_weeks, min_train_weeks=12)

    print(f"[rolling] planned_splits={len(splits)}")

    preds = []
    metrics_rows = []

    tuning_trials_all: list[pd.DataFrame] = []

    for split_idx, split in enumerate(track(splits, description="[rolling] splits"), start=1):
        train = model_df[model_df["week_start"] <= split.train_end]
        test = model_df[(model_df["week_start"] >= split.test_start) & (model_df["week_start"] <= split.test_end)]
        if train.empty or test.empty:
            print("[rolling] skip: empty train/test", flush=True)
            continue

        # Some early windows may have no positive labels when using time-aware thresholds.
        # Skip these splits instead of trying to fit a classifier on a single class.
        if train["label_high_risk"].nunique(dropna=False) < 2:
            print("[rolling] skip: single-class train window", flush=True)
            continue

        base_params = cfg.model.xgb.model_dump()

        # Inner tuning split (time-aware) to choose better hyperparameters without peeking at test.
        tuned_params = base_params
        tv = None
        if cfg.training.tuning.enabled:
            tv = time_based_train_val_split(
                train,
                week_col="week_start",
                validation_weeks=cfg.training.tuning.validation_weeks,
                label_col="label_high_risk",
            )
            if tv is not None:
                tuned_params, trials_df = tune_xgb_params(
                    fit_df=tv.fit_df,
                    val_df=tv.val_df,
                    label_col="label_high_risk",
                    feature_columns=feature_columns,
                    base_params=base_params,
                    n_trials=cfg.training.tuning.n_trials,
                    metric=cfg.training.tuning.metric,
                    random_state=cfg.training.tuning.random_state,
                    early_stopping_rounds=cfg.training.early_stopping_rounds,
                    verbose=True,
                    progress_label=f"split {split_idx}/{len(splits)}",
                )
                trials_df = trials_df.copy()
                trials_df["train_end"] = split.train_end
                trials_df["test_start"] = split.test_start
                trials_df["test_end"] = split.test_end
                tuning_trials_all.append(trials_df)

        # Train final model (use the split created for tuning if present; otherwise train on all train)
        fit_train = tv.fit_df if tv is not None else train
        val_for_es = tv.val_df if tv is not None else None

        xgb_res = train_xgb(
            train_df=fit_train,
            label_col="label_high_risk",
            feature_columns=feature_columns,
            params=tuned_params,
            eval_df=val_for_es,
            early_stopping_rounds=cfg.training.early_stopping_rounds,
        )

        model_for_pred = xgb_res.model
        if tv is not None and cfg.training.calibration.method != "none":
            model_for_pred = calibrate_prefit(
                model=xgb_res.model,
                val_df=tv.val_df,
                feature_columns=feature_columns,
                label_col="label_high_risk",
                method=cfg.training.calibration.method,
            )

        prob = predict_xgb(model_for_pred, test, feature_columns)
        pred = test[[cfg.boundaries_id_col, "week_start", "label_high_risk"]].copy()
        pred["risk_prob"] = prob
        preds.append(pred)

        y_true = pred["label_high_risk"].to_numpy().astype(int)
        y_prob = pred["risk_prob"].to_numpy().astype(float)
        y_hat = (y_prob >= cfg.thresholds.high).astype(int)
        hr, far = hit_false_alarm_rates(y_true, y_hat)

        metrics_rows.append(
            {
                "train_end": split.train_end,
                "test_start": split.test_start,
                "test_end": split.test_end,
                "auc": safe_auc(y_true, y_prob),
                "pr_auc": pr_auc(y_true, y_prob),
                "hit_rate": hr,
                "false_alarm_rate": far,
                "n_test": len(test),
            }
        )

    if not preds:
        raise RuntimeError("Not enough data to create rolling splits. Provide more weeks or lower min_train_weeks.")

    pred_df = pd.concat(preds, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(cfg.output_dir / "predictions.parquet", index=False)
    metrics_df.to_csv(cfg.output_dir / "rolling_metrics.csv", index=False)

    if tuning_trials_all:
        tuning_trials = pd.concat(tuning_trials_all, ignore_index=True)
        tuning_trials.to_csv(cfg.output_dir / "tuning_trials.csv", index=False)

    return pred_df


def run_end_to_end(cfg: AppConfig) -> None:
    cfg.validate_sources()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility artifact (config + environment snapshot)
    write_run_metadata(
        cfg.output_dir,
        config_dict=cfg.model_dump(mode="json"),
        input_files={
            "boundaries_path": str(cfg.boundaries_path) if cfg.boundaries_path is not None else None,
            "cases_csv": str(cfg.cases_csv),
            "climate_csv": str(cfg.climate_csv) if cfg.climate_csv is not None else None,
            "sanitation_csv": str(cfg.sanitation_csv) if cfg.sanitation_csv is not None else None,
            "flood_weekly_csv": str(cfg.flood_weekly_csv) if cfg.flood_weekly_csv is not None else None,
        },
    )

    # Build ward-week table
    ward_week, feature_columns = build_ward_week_table(cfg)
    ward_week.to_parquet(cfg.output_dir / "ward_week_features.parquet", index=False)

    # Train + rolling predictions
    pred_df = run_xgb_rolling(cfg, ward_week, feature_columns)

    # Apply thresholds to predicted risk
    pred_df2 = apply_risk_thresholds(
        pred_df,
        id_col=cfg.boundaries_id_col,
        prob_col="risk_prob",
        low=cfg.thresholds.low,
        high=cfg.thresholds.high,
        consecutive_weeks=cfg.thresholds.consecutive_weeks,
    )
    pred_df2.to_parquet(cfg.output_dir / "predictions_thresholded.parquet", index=False)

    # Fit on full data for SHAP explainability (global model)
    full = ward_week.dropna(subset=["label_high_risk"]).copy()
    xgb_res = train_xgb(
        train_df=full,
        label_col="label_high_risk",
        feature_columns=feature_columns,
        params=cfg.model.xgb.model_dump(),
    )

    X = full[feature_columns].fillna(0.0)
    save_shap_artifacts(xgb_res.model, X, cfg.output_dir / "shap", prefix="xgb")

    # GIS outputs (optional)
    if cfg.boundaries_path is not None:
        boundaries = read_boundaries(cfg.boundaries_path, cfg.boundaries_id_col)
        # Avoid writing a massive GeoJSON by repeating full geometries for every week.
        # Keep a compact snapshot for the latest available week.
        latest_week = pd.to_datetime(pred_df2["week_start"]).max()
        latest_pred = pred_df2[pred_df2["week_start"] == latest_week].copy()
        geo = boundaries.merge(latest_pred, on=cfg.boundaries_id_col, how="left")
        geo.to_file(cfg.output_dir / "risk_predictions.geojson", driver="GeoJSON")

        make_latest_risk_map(boundaries, pred_df2, cfg.boundaries_id_col, cfg.output_dir / "risk_latest_map.html")
        make_timeslider_map(boundaries, pred_df2, cfg.boundaries_id_col, cfg.output_dir / "risk_timeslider_map.html")

    # Finalize metadata with output hashes so edits are detectable.
    finalize_run_metadata(cfg.output_dir)
