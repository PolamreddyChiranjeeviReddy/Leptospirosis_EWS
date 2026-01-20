from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _fmt_int(x: int | float | None) -> str:
    if x is None:
        return "-"
    try:
        if pd.isna(x):
            return "-"
    except Exception:
        pass
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "-"
    try:
        if pd.isna(x):
            return "-"
    except Exception:
        pass
    try:
        return f"{100.0 * float(x):.1f}%"
    except Exception:
        return str(x)


def build_clinical_report(input_csv: Path, out_dir: Path) -> Path:
    input_csv = input_csv.resolve()
    out_dir = out_dir.resolve()
    _safe_mkdir(out_dir)

    # Codebook mappings (from Data_Gathered_By_Me/Code book.txt)
    hospital_map = {
        1: "Base Hospital Awissawella",
        2: "District General Hospital Kegalle",
        3: "Base hospital Karawanella",
        4: "District General Hospital Polonnaruwa",
        5: "Teaching Hospital Rathnapura",
        6: "Sri Jayawardanapura General Hospital",
        7: "Teaching Hospital Anuradhapura",
        8: "Teaching Hospital Peradeniya",
    }
    sex_map = {1: "Male", 2: "Female"}
    icu_map = {1: "Yes", 2: "No"}
    opd_map = {1: "OPD", 2: "Inward"}

    df = pd.read_csv(input_csv, low_memory=False)

    needed = ["Year", "Month", "Hospital", "Sex", "Age", "ICU", "OPD", "Final"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in clinical CSV: {missing}. "
            f"Found columns like: {list(df.columns[:20])}..."
        )

    # Convert key fields to numeric
    for col in [
        "Year",
        "Month",
        "Hospital",
        "Sex",
        "Age",
        "ICU",
        "OPD",
        "Final",
        "WPqPCRDiagnosis",
        "UrineqPCRDiagnosis",
        "SerumqPCRDiagnosis",
        "CultureqPCRDia",
        "UFqPCRDiag",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Treat 98/99 as missing for these coded fields
    coded_cols = [
        "Hospital",
        "Sex",
        "ICU",
        "OPD",
        "Final",
        "WPqPCRDiagnosis",
        "UrineqPCRDiagnosis",
        "SerumqPCRDiagnosis",
        "CultureqPCRDia",
        "UFqPCRDiag",
    ]
    for col in coded_cols:
        if col in df.columns:
            df.loc[df[col].isin([98, 99]), col] = np.nan

    # Age can also use 99 as missing in many datasets
    if "Age" in df.columns:
        df.loc[df["Age"].isin([98, 99]), "Age"] = np.nan

    # Construct a month-level date
    df["date"] = pd.to_datetime(
        {"year": df["Year"], "month": df["Month"], "day": 1}, errors="coerce"
    )

    df["hospital_name"] = df["Hospital"].map(hospital_map).fillna("Unknown")
    df["sex_label"] = df["Sex"].map(sex_map).fillna("Unknown")
    df["icu_label"] = df["ICU"].map(icu_map).fillna("Unknown")
    df["opd_label"] = df["OPD"].map(opd_map).fillna("Unknown")
    df["month_name"] = df["Month"].map(lambda m: calendar.month_name[int(m)] if pd.notna(m) else None)

    # Primary outcome: Final diagnosis (codebook says 1=Confirmed, 2=Not detected)
    df["final_confirmed"] = df["Final"].eq(1)

    # Secondary: any qPCR says Confirmed
    qpcr_cols = [c for c in ["WPqPCRDiagnosis", "UrineqPCRDiagnosis", "SerumqPCRDiagnosis", "CultureqPCRDia", "UFqPCRDiag"] if c in df.columns]
    if qpcr_cols:
        df["any_qpcr_confirmed"] = df[qpcr_cols].eq(1).any(axis=1)
    else:
        df["any_qpcr_confirmed"] = False

    # Basic sanity filtering
    df_valid_time = df.dropna(subset=["date"])

    # --- Aggregations
    by_month = (
        df_valid_time.groupby("date", as_index=False)
        .agg(
            n_records=("Serial", "count") if "Serial" in df_valid_time.columns else ("Year", "count"),
            n_final_confirmed=("final_confirmed", "sum"),
            n_any_qpcr_confirmed=("any_qpcr_confirmed", "sum"),
        )
        .sort_values("date")
    )
    by_month["final_confirmed_rate"] = by_month["n_final_confirmed"] / by_month["n_records"].replace(0, np.nan)

    by_hospital = (
        df_valid_time.groupby("hospital_name", as_index=False)
        .agg(
            n_records=("Year", "count"),
            n_final_confirmed=("final_confirmed", "sum"),
        )
        .sort_values("n_final_confirmed", ascending=False)
    )
    by_hospital["final_confirmed_rate"] = by_hospital["n_final_confirmed"] / by_hospital["n_records"].replace(0, np.nan)

    # Save a compact timeseries for reuse
    by_month.to_csv(out_dir / "clinical_monthly_counts.csv", index=False)
    by_hospital.to_csv(out_dir / "clinical_by_hospital.csv", index=False)

    # --- Plots
    # Plot 1: monthly confirmed counts
    plt.figure(figsize=(9, 4))
    plt.plot(by_month["date"], by_month["n_final_confirmed"], marker="o", linewidth=1)
    plt.title("Monthly confirmed leptospirosis cases (Final diagnosis)")
    plt.xlabel("Month")
    plt.ylabel("Confirmed cases")
    plt.grid(True, alpha=0.25)
    _save_fig(out_dir / "monthly_confirmed_cases.png")

    # Plot 2: confirmed rate over time
    plt.figure(figsize=(9, 4))
    plt.plot(by_month["date"], by_month["final_confirmed_rate"], marker="o", linewidth=1)
    plt.ylim(0, 1)
    plt.title("Monthly confirmed rate (Final confirmed / all records)")
    plt.xlabel("Month")
    plt.ylabel("Confirmed rate")
    plt.grid(True, alpha=0.25)
    _save_fig(out_dir / "monthly_confirmed_rate.png")

    # Plot 3: top hospitals by confirmed count
    top_h = by_hospital.head(10).copy()
    plt.figure(figsize=(9, 4.5))
    plt.barh(list(reversed(top_h["hospital_name"])), list(reversed(top_h["n_final_confirmed"])))
    plt.title("Top hospitals by confirmed count")
    plt.xlabel("Confirmed cases")
    plt.ylabel("Hospital")
    _save_fig(out_dir / "top_hospitals_confirmed.png")

    # Plot 4: age distribution
    age = df_valid_time[["Age", "final_confirmed"]].dropna(subset=["Age"]).copy()
    plt.figure(figsize=(9, 4))
    if not age.empty:
        bins = np.arange(0, max(100, int(np.nanmax(age["Age"]) + 5)), 5)
        confirmed_ages = age.loc[age["final_confirmed"], "Age"].to_numpy()
        not_ages = age.loc[~age["final_confirmed"], "Age"].to_numpy()
        plt.hist(not_ages, bins=bins, alpha=0.55, label="Not detected")
        plt.hist(confirmed_ages, bins=bins, alpha=0.55, label="Confirmed")
        plt.title("Age distribution (Final diagnosis)")
        plt.xlabel("Age (years)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, axis="y", alpha=0.25)
    _save_fig(out_dir / "age_distribution.png")

    # Plot 5: ICU proportion among confirmed
    icu = df_valid_time.loc[df_valid_time["final_confirmed"], ["icu_label"]].copy()
    icu_counts = icu["icu_label"].value_counts().reindex(["Yes", "No", "Unknown"]).fillna(0)
    plt.figure(figsize=(6, 4))
    plt.bar(icu_counts.index, icu_counts.values)
    plt.title("ICU admission among confirmed")
    plt.xlabel("ICU")
    plt.ylabel("Count")
    _save_fig(out_dir / "icu_among_confirmed.png")

    # --- HTML report
    n_total = int(len(df))
    n_with_time = int(len(df_valid_time))
    n_confirmed = int(df_valid_time["final_confirmed"].sum())
    confirmed_rate = (n_confirmed / n_with_time) if n_with_time else None

    date_min = df_valid_time["date"].min()
    date_max = df_valid_time["date"].max()

    html: list[str] = []
    html.append("<html><head><meta charset='utf-8'/><title>Clinical Dataset Report</title>")
    html.append(
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;}h1,h2{margin:0.2em 0;}"
        "table{border-collapse:collapse;margin:12px 0;}th,td{border:1px solid #ddd;padding:6px 8px;font-size:13px;}"
        "th{background:#f6f6f6;text-align:left;} .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}"
        "img{max-width:100%;border:1px solid #eee;} .note{color:#444;background:#fafafa;border:1px solid #eee;padding:10px;}"
        "</style></head><body>"
    )

    html.append("<h1>Leptospirosis Clinical Dataset — Summary Report</h1>")
    html.append(f"<div class='note'>Input file: <b>{input_csv}</b><br/>Output folder: <b>{out_dir}</b></div>")

    html.append("<h2>1) Key numbers</h2>")
    html.append("<table><tr><th>Metric</th><th>Value</th></tr>")
    rows = [
        ("Total rows (raw)", _fmt_int(n_total)),
        ("Rows with valid Year/Month", _fmt_int(n_with_time)),
        ("Date range", f"{date_min.date() if pd.notna(date_min) else '-'} → {date_max.date() if pd.notna(date_max) else '-'}"),
        ("Final confirmed", _fmt_int(n_confirmed)),
        ("Final confirmed rate", _fmt_pct(confirmed_rate)),
        ("Hospitals (unique)", _fmt_int(int(df_valid_time["hospital_name"].nunique()))),
    ]
    for k, v in rows:
        html.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
    html.append("</table>")

    html.append("<h2>2) Plots</h2>")
    html.append("<div class='grid'>")
    for title, fname in [
        ("Monthly confirmed cases", "monthly_confirmed_cases.png"),
        ("Monthly confirmed rate", "monthly_confirmed_rate.png"),
        ("Top hospitals (confirmed)", "top_hospitals_confirmed.png"),
        ("Age distribution", "age_distribution.png"),
        ("ICU among confirmed", "icu_among_confirmed.png"),
    ]:
        html.append(f"<div><h3>{title}</h3><img src='{fname}'/></div>")
    html.append("</div>")

    html.append("<h2>3) Top hospitals table</h2>")
    html.append("<table><tr><th>Hospital</th><th>Records</th><th>Confirmed</th><th>Confirmed rate</th></tr>")
    for _, r in by_hospital.head(12).iterrows():
        html.append(
            "<tr>"
            f"<td>{r['hospital_name']}</td>"
            f"<td>{_fmt_int(r['n_records'])}</td>"
            f"<td>{_fmt_int(r['n_final_confirmed'])}</td>"
            f"<td>{_fmt_pct(float(r['final_confirmed_rate']) if pd.notna(r['final_confirmed_rate']) else None)}</td>"
            "</tr>"
        )
    html.append("</table>")

    html.append("<h2>4) Notes</h2>")
    html.append(
        "<div class='note'>"
        "This report uses <b>Final diagnosis</b> (codebook: 1=Confirmed, 2=Not detected) as the main outcome. "
        "Values like 98/99 are treated as missing for coded fields. "
        "Because this dataset is patient-level (not ward-week counts), it is best used for clinical/descriptive reporting, "
        "or for building a hospital-level time series if you later want to connect it to weather/flood data." 
        "</div>"
    )

    html.append("</body></html>")

    report_path = out_dir / "clinical_report.html"
    report_path.write_text("\n".join(html), encoding="utf-8")
    return report_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_csv = (project_root.parent / "Data_Gathered_By_Me" / "Leptospirosis clinical data.csv").resolve()

    parser = argparse.ArgumentParser(description="Generate an HTML report from the clinical leptospirosis dataset")
    parser.add_argument("--input-csv", type=Path, default=default_csv, help="Path to the clinical CSV")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(project_root / "outputs" / "clinical_report"),
        help="Output folder for report + plots",
    )
    args = parser.parse_args()

    report_path = build_clinical_report(args.input_csv, args.out_dir)
    print(f"Wrote clinical report: {report_path}")


if __name__ == "__main__":
    main()
