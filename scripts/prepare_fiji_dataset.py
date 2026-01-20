from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fiji-weekly-csv",
        required=True,
        help="Path to leptospirosis_Fiji_2023 weeklyCaseDate.csv",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory where prepared CSVs will be written",
    )
    args = ap.parse_args()

    src = Path(args.fiji_weekly_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    required = {"date", "division", "cases"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in Fiji CSV: {sorted(missing)}")

    # Fiji file uses dd/mm/yyyy (e.g., 02/01/2006 == 2 Jan 2006)
    dts = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    if dts.isna().any():
        bad = int(dts.isna().sum())
        raise SystemExit(f"Failed to parse {bad} dates from Fiji CSV")

    cases = pd.DataFrame(
        {
            "ward_id": df["division"].astype(str).str.strip().str.lower(),
            "date": dts.dt.strftime("%Y-%m-%d"),
            "cases": pd.to_numeric(df["cases"], errors="coerce").fillna(0.0).astype(int),
        }
    )

    cases_path = out_dir / "fiji_cases.csv"
    cases.to_csv(cases_path, index=False)

    # Optional static table (not scientifically meaningful here, but keeps the pipeline happy
    # if you choose to provide sanitation_csv). We set sanitation components to 0.
    if "pop" in df.columns:
        pop = (
            df[["division", "pop"]]
            .dropna()
            .groupby("division", as_index=False)["pop"]
            .max()
            .rename(columns={"division": "ward_id", "pop": "pop_density"})
        )
        pop["ward_id"] = pop["ward_id"].astype(str).str.strip().str.lower()
    else:
        pop = pd.DataFrame({"ward_id": sorted(cases["ward_id"].unique()), "pop_density": 0.0})

    sanitation = pop.copy()
    sanitation["open_drain_pct"] = 0.0
    sanitation["no_toilet_pct"] = 0.0
    sanitation["unsafe_water_pct"] = 0.0

    san_path = out_dir / "fiji_sanitation_placeholder.csv"
    sanitation.to_csv(san_path, index=False)

    print("Wrote:")
    print("-", cases_path)
    print("-", san_path)


if __name__ == "__main__":
    main()
