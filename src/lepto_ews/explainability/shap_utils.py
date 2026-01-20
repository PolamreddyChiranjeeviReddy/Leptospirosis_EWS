from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def save_shap_artifacts(model, X: pd.DataFrame, output_dir: Path, prefix: str = "xgb") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Save values
    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in X.columns])
    shap_df.to_parquet(output_dir / f"{prefix}_shap_values.parquet", index=False)

    # Global summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_shap_summary.png", dpi=200)
    plt.close()

    # Global bar plot
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_shap_importance.png", dpi=200)
    plt.close()


def top_reason_text(shap_row: np.ndarray, feature_names: list[str], k: int = 3) -> str:
    idx = np.argsort(np.abs(shap_row))[::-1][:k]
    parts = [feature_names[i] for i in idx]
    return ", ".join(parts)
