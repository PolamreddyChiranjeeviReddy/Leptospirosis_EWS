from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class _CalibratedModel:
    base_model: object
    method: str
    sigmoid_model: LogisticRegression | None = None
    isotonic_model: IsotonicRegression | None = None

    def predict_proba(self, X):
        p = np.asarray(self.base_model.predict_proba(X))[:, 1].astype(float)
        p = np.clip(p, 1e-6, 1 - 1e-6)

        if self.method == "sigmoid":
            if self.sigmoid_model is None:
                raise RuntimeError("Sigmoid calibrator not fitted")
            logit = np.log(p / (1.0 - p)).reshape(-1, 1)
            p_cal = np.asarray(self.sigmoid_model.predict_proba(logit))[:, 1]
        elif self.method == "isotonic":
            if self.isotonic_model is None:
                raise RuntimeError("Isotonic calibrator not fitted")
            p_cal = np.asarray(self.isotonic_model.predict(p))
        else:
            p_cal = p

        p_cal = np.clip(p_cal, 0.0, 1.0)
        return np.column_stack([1.0 - p_cal, p_cal])


def calibrate_prefit(
    *,
    model,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    label_col: str,
    method: str,
):
    """Calibrate a prefit classifier on a held-out validation set.

    Returns a model-like object with predict_proba.
    """
    if method == "none":
        return model

    y = val_df[label_col].to_numpy().astype(int)
    if len(np.unique(y)) < 2:
        return model

    if method not in {"sigmoid", "isotonic"}:
        return model

    X = val_df[feature_columns].to_numpy()
    p = np.asarray(model.predict_proba(X))[:, 1].astype(float)
    p = np.clip(p, 1e-6, 1 - 1e-6)

    try:
        if method == "sigmoid":
            logit = np.log(p / (1.0 - p)).reshape(-1, 1)
            lr = LogisticRegression(max_iter=1000)
            lr.fit(logit, y)
            return _CalibratedModel(base_model=model, method=method, sigmoid_model=lr)

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p, y)
        return _CalibratedModel(base_model=model, method=method, isotonic_model=iso)
    except Exception:
        # If calibration fails for any reason, fall back to the base model.
        return model
