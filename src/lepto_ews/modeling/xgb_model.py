from __future__ import annotations

from dataclasses import dataclass
import inspect

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


@dataclass
class XGBResult:
    model: XGBClassifier
    feature_columns: list[str]


def _with_scale_pos_weight(params: dict, y: np.ndarray) -> dict:
    """Add scale_pos_weight if not provided, to help with class imbalance."""
    params = dict(params)
    if params.get("scale_pos_weight") is not None:
        return params

    y = np.asarray(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos <= 0 or n_neg <= 0:
        return params

    params["scale_pos_weight"] = float(n_neg / n_pos)
    return params


def train_xgb(
    train_df: pd.DataFrame,
    label_col: str,
    feature_columns: list[str],
    params: dict,
    *,
    eval_df: pd.DataFrame | None = None,
    early_stopping_rounds: int | None = None,
) -> XGBResult:
    X = train_df[feature_columns].to_numpy()
    y = train_df[label_col].to_numpy().astype(int)

    params = _with_scale_pos_weight(params, y)

    model = XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="aucpr",
        n_jobs=0,
    )

    fit_kwargs: dict = {}
    if eval_df is not None:
        X_val = eval_df[feature_columns].to_numpy()
        y_val = eval_df[label_col].to_numpy().astype(int)
        if len(np.unique(y_val)) >= 2:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False

            if early_stopping_rounds is not None and early_stopping_rounds > 0:
                fit_params = inspect.signature(model.fit).parameters
                if "early_stopping_rounds" in fit_params:
                    fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)

    model.fit(X, y, **fit_kwargs)
    return XGBResult(model=model, feature_columns=feature_columns)


def predict_xgb(model, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    X = df[feature_columns].to_numpy()
    return model.predict_proba(X)[:, 1]
