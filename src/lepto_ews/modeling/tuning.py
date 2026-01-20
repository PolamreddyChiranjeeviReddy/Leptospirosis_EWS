from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lepto_ews.modeling.metrics import pr_auc, safe_auc
from lepto_ews.modeling.xgb_model import predict_xgb, train_xgb


@dataclass(frozen=True)
class TrainValSplit:
    fit_df: pd.DataFrame
    val_df: pd.DataFrame


def time_based_train_val_split(
    train_df: pd.DataFrame,
    *,
    week_col: str,
    validation_weeks: int,
    label_col: str | None = None,
    min_fit_weeks: int = 8,
) -> TrainValSplit | None:
    weeks = sorted(pd.to_datetime(train_df[week_col].unique()))
    if validation_weeks <= 0:
        return None
    if len(weeks) < (validation_weeks + min_fit_weeks):
        return None

    val_weeks = set(weeks[-validation_weeks:])
    fit = train_df[~pd.to_datetime(train_df[week_col]).isin(val_weeks)].copy()
    val = train_df[pd.to_datetime(train_df[week_col]).isin(val_weeks)].copy()

    if fit.empty or val.empty:
        return None

    if label_col is not None:
        fit_classes = fit[label_col].dropna().unique()
        val_classes = val[label_col].dropna().unique()
        if len(fit_classes) < 2 or len(val_classes) < 2:
            return None

    return TrainValSplit(fit_df=fit, val_df=val)


def _sample_params(rng: np.random.Generator, base_params: dict) -> dict:
    """Sample a single param set around a reasonable XGBoost search space."""
    p = dict(base_params)

    p["max_depth"] = int(rng.integers(3, 8))
    p["min_child_weight"] = float(rng.choice([1.0, 2.0, 5.0, 10.0]))

    p["learning_rate"] = float(rng.choice([0.01, 0.03, 0.05, 0.08, 0.1]))
    p["subsample"] = float(rng.uniform(0.6, 1.0))
    p["colsample_bytree"] = float(rng.uniform(0.6, 1.0))

    # regularization
    p["reg_lambda"] = float(10 ** rng.uniform(-0.3, 0.8))  # ~[0.5, 6.3]
    p["reg_alpha"] = float(10 ** rng.uniform(-3.0, -0.3))  # ~[0.001, 0.5]
    p["gamma"] = float(rng.choice([0.0, 0.05, 0.1, 0.2, 0.4]))

    # allow more trees; early stopping will cap it
    p["n_estimators"] = int(max(p.get("n_estimators", 500), 800))

    return p


def tune_xgb_params(
    *,
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_col: str,
    feature_columns: list[str],
    base_params: dict,
    n_trials: int,
    metric: str,
    random_state: int,
    early_stopping_rounds: int | None,
    verbose: bool = False,
    progress_label: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Random search tuning using a time-based validation split.

    Returns (best_params, trials_df)
    """
    rng = np.random.default_rng(random_state)

    results: list[dict] = []
    best_score = -np.inf
    best_params = dict(base_params)

    # Tuning requires both classes in both datasets. If not present, skip tuning.
    fit_classes = fit_df[label_col].dropna().unique()
    val_classes = val_df[label_col].dropna().unique()
    if len(fit_classes) < 2 or len(val_classes) < 2:
        return best_params, pd.DataFrame([])

    total = int(max(1, n_trials))
    for i in range(total):
        if verbose:
            label = f" {progress_label}" if progress_label else ""
            msg = f"[tuning{label}] trial {i + 1}/{total}"
            print(msg, end="\r", flush=True)
        params = _sample_params(rng, base_params)

        res = train_xgb(
            train_df=fit_df,
            label_col=label_col,
            feature_columns=feature_columns,
            params=params,
            eval_df=val_df,
            early_stopping_rounds=early_stopping_rounds,
        )
        y_true = val_df[label_col].to_numpy().astype(int)
        y_prob = predict_xgb(res.model, val_df, feature_columns).astype(float)

        auc_score = safe_auc(y_true, y_prob)
        pr_score = pr_auc(y_true, y_prob)

        score = pr_score if metric == "pr_auc" else auc_score

        results.append(
            {
                "trial": i + 1,
                "score": score,
                "auc": auc_score,
                "pr_auc": pr_score,
                "max_depth": params.get("max_depth"),
                "learning_rate": params.get("learning_rate"),
                "subsample": params.get("subsample"),
                "colsample_bytree": params.get("colsample_bytree"),
                "min_child_weight": params.get("min_child_weight"),
                "reg_lambda": params.get("reg_lambda"),
                "reg_alpha": params.get("reg_alpha"),
                "gamma": params.get("gamma"),
                "n_estimators": params.get("n_estimators"),
            }
        )

        if np.isfinite(score) and score > best_score:
            best_score = float(score)
            best_params = params

    if verbose:
        # Clear the progress line.
        print(" " * 80, end="\r", flush=True)

    trials_df = pd.DataFrame(results).sort_values("score", ascending=False)
    return best_params, trials_df
