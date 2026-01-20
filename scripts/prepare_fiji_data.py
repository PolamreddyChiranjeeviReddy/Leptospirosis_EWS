from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default=str(Path("..") / "leptospirosis_Fiji_2023-main" / "data" / "weeklyCaseDate.csv"),
        help="Path to Fiji weeklyCaseDate.csv",
    )
    ap.add_argument(
        "--out-dir",
        default=str(Path("data") / "fiji"),
        help="Output directory (will write cases.csv and climate.csv)",
    )
    args = ap.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    required = {"division", "date", "cases"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Fiji weeklyCaseDate.csv missing columns: {sorted(missing)}")

    # Parse date: file uses dd/mm/yyyy
    df = df[["division", "date", "cases"]].copy()
    df["division"] = df["division"].astype(str).str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    if df["date"].isna().any():
        bad = int(df["date"].isna().sum())
        raise SystemExit(f"Failed to parse {bad} date rows in {src}")

    df["cases"] = pd.to_numeric(df["cases"], errors="coerce").fillna(0).astype(int)

    # EWS expects: id_col, date, cases
    cases = df.rename(columns={"division": "division_id"})
    cases.to_csv(out_dir / "cases.csv", index=False)

    # Fiji CSV doesn't include climate in a CSV-friendly way (climate is in .rds).
    # Write a minimal climate file so the pipeline can run; the model will rely on lagged cases + seasonality.
    climate = cases[["division_id", "date"]].copy()
    climate["rain_mm"] = 0.0
    climate.to_csv(out_dir / "climate.csv", index=False)

    print("Wrote:")
    print(out_dir / "cases.csv")
    print(out_dir / "climate.csv")


if __name__ == "__main__":
    main()
