from __future__ import annotations

import argparse
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lepto_ews.run_metadata import finalize_run_metadata


def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(float)
    y_prob = y_prob.astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


def _save_calibration_plot(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    bins = np.linspace(0.0, 1.0, 11)
    idx = np.digitize(y_prob, bins) - 1
    xs = []
    ys = []
    ns = []
    for b in range(10):
        mask = idx == b
        if not np.any(mask):
            continue
        xs.append(float(np.mean(y_prob[mask])))
        ys.append(float(np.mean(y_true[mask])))
        ns.append(int(np.sum(mask)))

    plt.figure(figsize=(5.5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect")
    plt.plot(xs, ys, marker="o", linewidth=1.5, label="Model")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.title("Calibration (reliability curve)")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction positive")
    plt.grid(True, alpha=0.25)
    plt.legend()
    _save_fig(out_path)


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _fmt_float(x: float | int | None) -> str:
    if x is None:
        return "-"
    try:
        if pd.isna(x):
            return "-"
    except Exception:
        pass
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def build_report(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    report_dir = run_dir / "report"
    _safe_mkdir(report_dir)

    # Inputs
    rolling_csv = run_dir / "rolling_metrics.csv"
    pred_path = run_dir / "predictions.parquet"
    pred_thr_path = run_dir / "predictions_thresholded.parquet"
    shap_values_path = run_dir / "shap" / "xgb_shap_values.parquet"
    tuning_trials_path = run_dir / "tuning_trials.csv"
    run_meta_path = run_dir / "run_metadata.json"

    # Determine the area id column (ward_id / district_id / division_id, etc.)
    id_col = None
    horizon_weeks: int | None = None
    if run_meta_path.exists():
        try:
            meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
            cfg = meta.get("config") or {}
            id_col = cfg.get("boundaries_id_col")
            horizon_weeks = cfg.get("prediction_horizon_weeks")
        except Exception:
            id_col = None

    metrics = pd.read_csv(rolling_csv, parse_dates=["train_end", "test_start", "test_end"])
    pred = pd.read_parquet(pred_path)
    pred_thr = pd.read_parquet(pred_thr_path)

    # --- Export easy-to-share CSVs (so faculty can open in Excel)
    # Keep them under report/ so they are served alongside the HTML.
    pred_csv_path = report_dir / "predictions.csv"
    pred_thr_csv_path = report_dir / "predictions_thresholded.csv"
    pred_latest_csv_path = report_dir / "predictions_latest_week.csv"
    pred_thr_latest_csv_path = report_dir / "predictions_thresholded_latest_week.csv"

    # Full exports
    pred.to_csv(pred_csv_path, index=False)
    pred_thr.to_csv(pred_thr_csv_path, index=False)

    # Latest-week exports
    try:
        latest_week = pd.to_datetime(pred["week_start"]).max()
        pred_latest = pred[pd.to_datetime(pred["week_start"]) == latest_week].copy()
        pred_latest.to_csv(pred_latest_csv_path, index=False)

        latest_week_thr = pd.to_datetime(pred_thr["week_start"]).max()
        pred_thr_latest = pred_thr[pd.to_datetime(pred_thr["week_start"]) == latest_week_thr].copy()
        pred_thr_latest.to_csv(pred_thr_latest_csv_path, index=False)
    except Exception:
        # If parsing fails for some reason, skip latest-week exports.
        pass

    # If id_col wasn't found from metadata, infer from prediction columns.
    if id_col is None or id_col not in pred.columns:
        candidates = [c for c in pred.columns if c.endswith("_id")]
        if "ward_id" in pred.columns:
            id_col = "ward_id"
        elif len(candidates) == 1:
            id_col = candidates[0]
        elif len(candidates) > 1:
            id_col = candidates[0]
        else:
            id_col = None

    if id_col is None or id_col not in pred.columns:
        candidates_thr = [c for c in pred_thr.columns if c.endswith("_id")]
        if len(candidates_thr) >= 1:
            id_col = candidates_thr[0]
        else:
            id_col = pred.columns[0]

    # --- Plot 0: calibration curve (overall)
    cal_plot_path = report_dir / "calibration_curve.png"
    y_true_all = pred["label_high_risk"].to_numpy().astype(int)
    y_prob_all = pred["risk_prob"].to_numpy().astype(float)
    brier = _brier_score(y_true_all, y_prob_all)
    _save_calibration_plot(y_true_all, y_prob_all, cal_plot_path)

    # --- Plot 1: Rolling AUC / PR-AUC over time
    plt.figure(figsize=(9, 4))
    m = metrics.sort_values("test_start")
    plt.plot(m["test_start"], m["auc"], marker="o", linewidth=1, label="AUC")
    plt.plot(m["test_start"], m["pr_auc"], marker="o", linewidth=1, label="PR-AUC")
    plt.ylim(0, 1)
    plt.title("Rolling validation performance (time-aware)")
    plt.xlabel("Test week")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.25)
    plt.legend()
    _save_fig(report_dir / "rolling_performance.png")

    # --- Plot 2: risk probability distribution
    plt.figure(figsize=(7, 4))
    plt.hist(pred["risk_prob"].to_numpy(), bins=20, edgecolor="white")
    plt.title("Predicted risk probability distribution")
    plt.xlabel("risk_prob")
    plt.ylabel("Count")
    plt.grid(True, axis="y", alpha=0.25)
    _save_fig(report_dir / "risk_prob_hist.png")

    # --- Plot 3: risk category counts
    plt.figure(figsize=(6, 4))
    counts = pred_thr["risk_category"].value_counts().reindex(["low", "medium", "high"]).fillna(0)
    plt.bar(counts.index, counts.values)
    plt.title("Final risk category counts (after consecutive-week rule)")
    plt.xlabel("risk_category")
    plt.ylabel("Count")
    _save_fig(report_dir / "risk_category_counts.png")

    # --- Plot 4: Top SHAP features (mean |SHAP|)
    top_shap_rows = []
    shap_plot_path = None
    if shap_values_path.exists():
        shap_df = pd.read_parquet(shap_values_path)
        shap_cols = [c for c in shap_df.columns if c.startswith("shap_")]
        if shap_cols:
            mean_abs = shap_df[shap_cols].abs().mean().sort_values(ascending=False)
            top = mean_abs.head(10)
            plt.figure(figsize=(9, 4.5))
            plt.barh(list(reversed(top.index)), list(reversed(top.values)))
            plt.title("Top drivers (mean |SHAP|)")
            plt.xlabel("Mean |SHAP value|")
            shap_plot_path = report_dir / "top_shap_features.png"
            _save_fig(shap_plot_path)

            top_shap_rows = [(k.replace("shap_", ""), float(v)) for k, v in top.items()]

    # --- Tables for report
    # Summary stats
    auc_nonnull = metrics["auc"].dropna()
    pr_nonnull = metrics["pr_auc"].dropna()

    summary = {
        "n_splits": int(len(metrics)),
        "n_splits_auc_defined": int(auc_nonnull.shape[0]),
        "mean_auc": float(auc_nonnull.mean()) if len(auc_nonnull) else None,
        "mean_pr_auc": float(pr_nonnull.mean()) if len(pr_nonnull) else None,
        "brier": brier,
        "risk_prob_min": float(pred["risk_prob"].min()),
        "risk_prob_max": float(pred["risk_prob"].max()),
        "n_prediction_rows": int(len(pred)),
        "n_areas": int(pred[id_col].nunique()) if (id_col is not None and id_col in pred.columns) else None,
        "weeks_start": str(pd.to_datetime(pred["week_start"]).min().date()),
        "weeks_end": str(pd.to_datetime(pred["week_start"]).max().date()),
    }

    tuning_summary_rows = []
    if tuning_trials_path.exists():
        try:
            tt = pd.read_csv(tuning_trials_path)
            # best trial per split (train_end)
            if "train_end" in tt.columns and "score" in tt.columns:
                best_each = tt.sort_values(["train_end", "score"], ascending=[True, False]).groupby("train_end").head(1)
                tuning_summary_rows = (
                    best_each[["train_end", "score", "auc", "pr_auc", "max_depth", "learning_rate", "subsample", "colsample_bytree"]]
                    .sort_values("train_end")
                    .to_dict("records")
                )
        except Exception:
            tuning_summary_rows = []

    # Top 10 highest-risk predictions
    top_risk = (
        pred_thr.sort_values("risk_prob", ascending=False)
        .head(10)[[id_col, "week_start", "risk_prob", "risk_category_raw", "risk_category"]]
        .copy()
    )
    top_risk["week_start"] = pd.to_datetime(top_risk["week_start"]).dt.date.astype(str)

    # Explain the consecutive-week rule impact (raw vs final)
    raw_counts = pred_thr["risk_category_raw"].value_counts().to_dict()
    final_counts = pred_thr["risk_category"].value_counts().to_dict()

    # --- Write HTML report
    latest_map = run_dir / "risk_latest_map.html"
    timeslider_map = run_dir / "risk_timeslider_map.html"

    html = []
    html.append("<html><head><meta charset='utf-8'/><title>EWS Results Report</title>")
    html.append(
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;}h1,h2{margin:0.2em 0;}"
        "table{border-collapse:collapse;margin:12px 0;}th,td{border:1px solid #ddd;padding:6px 8px;font-size:13px;}"
        "th{background:#f6f6f6;text-align:left;} .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}"
        "img{max-width:100%;border:1px solid #eee;} .note{color:#444;background:#fafafa;border:1px solid #eee;padding:10px;}"
        "</style></head><body>"
    )

    html.append("<h1>Leptospirosis EWS — Results Report</h1>")
    html.append(f"<div class='note'>Run folder: <b>{run_dir}</b></div>")
    if horizon_weeks is not None:
        html.append(
            f"<div class='note'><b>Prediction horizon:</b> {horizon_weeks} week(s) ahead. "
            "(Each row in predictions refers to the target <code>week_start</code>.)</div>"
        )

    html.append("<h2>1) What this run produced</h2>")
    html.append("<ul>")
    if horizon_weeks is not None:
        html.append(f"<li>Prediction horizon: <b>{horizon_weeks}</b> week(s) ahead</li>")
    html.append(
        f"<li>Predictions: <a href='../{pred_path.name}'><code>{pred_path.name}</code></a> "
        f"({summary['n_prediction_rows']} rows)</li>"
    )
    html.append(
        "<li>Predictions (CSV for Excel): "
        "<a href='predictions.csv'><code>report/predictions.csv</code></a> "
        "| <a href='predictions_latest_week.csv'><code>report/predictions_latest_week.csv</code></a></li>"
    )
    html.append(
        f"<li>Risk categories (thresholding + consecutive-week rule): "
        f"<a href='../{pred_thr_path.name}'><code>{pred_thr_path.name}</code></a></li>"
    )
    html.append(
        "<li>Risk categories (CSV for Excel): "
        "<a href='predictions_thresholded.csv'><code>report/predictions_thresholded.csv</code></a> "
        "| <a href='predictions_thresholded_latest_week.csv'><code>report/predictions_thresholded_latest_week.csv</code></a></li>"
    )
    html.append(
        f"<li>Rolling validation metrics: <a href='../{rolling_csv.name}'><code>{rolling_csv.name}</code></a> "
        f"({summary['n_splits']} splits)</li>"
    )
    if tuning_trials_path.exists():
        html.append(
            f"<li>Tuning trials: <a href='../{tuning_trials_path.name}'><code>{tuning_trials_path.name}</code></a></li>"
        )
    if run_meta_path.exists():
        html.append(
            f"<li>Reproducibility: <a href='../{run_meta_path.name}'><code>{run_meta_path.name}</code></a> "
            "(environment + config snapshot)</li>"
        )
    if latest_map.exists() and timeslider_map.exists():
        html.append(
            f"<li>Maps: <a href='../{latest_map.name}'><code>{latest_map.name}</code></a>, "
            f"<a href='../{timeslider_map.name}'><code>{timeslider_map.name}</code></a></li>"
        )
    if shap_values_path.exists():
        html.append(f"<li>Explainability: <a href='../shap/'><code>shap/</code></a> artifacts (SHAP)</li>")
    html.append("</ul>")

    html.append("<h2>2) Key summary numbers (easy to present)</h2>")
    html.append("<table>")
    html.append("<tr><th>Metric</th><th>Value</th></tr>")
    rows = [
        (f"Areas ({id_col or 'id'})", summary["n_areas"] if summary["n_areas"] is not None else "-"),
        ("Prediction weeks", f"{summary['weeks_start']} → {summary['weeks_end']}"),
        ("Risk probability range", f"{summary['risk_prob_min']:.4f} → {summary['risk_prob_max']:.4f}"),
        ("Rolling splits", summary["n_splits"]),
        (
            "Splits where AUC defined",
            f"{summary['n_splits_auc_defined']} (NaN happens when test window has only one class)",
        ),
        ("Mean AUC (defined splits)", _fmt_float(summary["mean_auc"])),
        ("Mean PR-AUC (defined splits)", _fmt_float(summary["mean_pr_auc"])),
        ("Brier score (lower is better)", _fmt_float(summary["brier"])),
        ("Raw category counts", str(raw_counts)),
        ("Final category counts", str(final_counts)),
    ]
    for k, v in rows:
        html.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
    html.append("</table>")

    html.append("<h2>3) Plots (screenshots for mentor)</h2>")
    html.append("<div class='grid'>")
    html.append("<div><h3>Rolling performance</h3><img src='rolling_performance.png'/></div>")
    html.append("<div><h3>Risk probability distribution</h3><img src='risk_prob_hist.png'/></div>")
    html.append("<div><h3>Calibration curve</h3><img src='calibration_curve.png'/></div>")
    html.append("<div><h3>Final risk category counts</h3><img src='risk_category_counts.png'/></div>")
    if shap_plot_path is not None:
        html.append("<div><h3>Top drivers (SHAP)</h3><img src='top_shap_features.png'/></div>")
    html.append("</div>")

    if tuning_summary_rows:
        html.append("<h2>3b) Tuning summary (best trial per split)</h2>")
        html.append("<table>")
        cols = list(tuning_summary_rows[0].keys())
        html.append("<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>")
        for r in tuning_summary_rows:
            html.append("<tr>" + "".join(f"<td>{r.get(c,'')}</td>" for c in cols) + "</tr>")
        html.append("</table>")

    html.append("<h2>4) Top 10 highest-risk predictions (examples)</h2>")
    html.append("<table>")
    html.append("<tr>" + "".join(f"<th>{c}</th>" for c in top_risk.columns) + "</tr>")
    for _, r in top_risk.iterrows():
        html.append("<tr>" + "".join(f"<td>{r[c]}</td>" for c in top_risk.columns) + "</tr>")
    html.append("</table>")

    if top_shap_rows:
        html.append("<h2>5) Top SHAP drivers (table)</h2>")
        html.append("<table><tr><th>Feature</th><th>Mean |SHAP|</th></tr>")
        for feat_name, val in top_shap_rows:
            html.append(f"<tr><td>{feat_name}</td><td>{val:.6f}</td></tr>")
        html.append("</table>")

    html.append("<h2>6) Maps</h2>")
    if latest_map.exists() and timeslider_map.exists():
        html.append(
            "<div class='note'>Open these files from the run folder, or serve the folder and open via browser.</div>"
        )
        html.append(
            "<div class='note'>Recommended (Windows): <code>python -m http.server 8000 --bind 127.0.0.1</code> from the project root, then open these links.</div>"
        )
        html.append("<ul>")
        html.append(
            f"<li>Latest week map: <a href='../{latest_map.name}' target='_blank' rel='noopener noreferrer'><code>{latest_map.name}</code></a></li>"
        )
        html.append(
            f"<li>Time slider map: <a href='../{timeslider_map.name}' target='_blank' rel='noopener noreferrer'><code>{timeslider_map.name}</code></a></li>"
        )
        html.append("</ul>")
    else:
        html.append("<div class='note'>Map HTML files not found in this run directory.</div>")

    html.append("</body></html>")

    report_path = report_dir / "results_report.html"
    report_path.write_text("\n".join(html), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a clear results report from an EWS run directory")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/sample_run"),
        help="Path to an outputs run directory (e.g., outputs/sample_run)",
    )
    args = parser.parse_args()
    report_path = build_report(args.run_dir)
    print(f"Wrote report: {report_path}")

    # The pipeline finalizes output hashes at the end of the run, but the report
    # is generated afterward. Refresh the output manifest/stamp so provenance
    # checks remain valid after report generation.
    try:
        finalize_run_metadata(args.run_dir)
    except Exception as e:
        print(f"WARNING: could not update run metadata hashes after report generation: {e}")


if __name__ == "__main__":
    main()
