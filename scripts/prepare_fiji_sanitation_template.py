"""Create a sanitation/WASH CSV template for Fiji divisions.

Because truly open subnational sanitation datasets are not always available,
this template lets you:
- start with population density (easy open proxy)
- later fill in open_drain_pct / no_toilet_pct / unsafe_water_pct from a
  survey or government report.

Output schema matches lepto_ews.io.read_sanitation_csv requirements.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare sanitation.csv template")
    parser.add_argument("--boundaries", type=Path, default=Path("data/fiji/fiji_adm1_divisions.gpkg"))
    parser.add_argument("--id-col", type=str, default="division_id")
    parser.add_argument("--out", type=Path, default=Path("data/fiji/sanitation.csv"))
    args = parser.parse_args()

    gdf = gpd.read_file(args.boundaries)
    if args.id_col not in gdf.columns:
        raise SystemExit(f"Missing id column '{args.id_col}' in {args.boundaries}. Columns: {list(gdf.columns)}")

    ids = sorted(gdf[args.id_col].astype(str).unique().tolist())
    df = pd.DataFrame(
        {
            args.id_col: ids,
            # Required WASH columns (fill later if you find real subnational sources)
            "open_drain_pct": ["" for _ in ids],
            "no_toilet_pct": ["" for _ in ids],
            "unsafe_water_pct": ["" for _ in ids],
            # Optional column supported by the pipeline
            "pop_density": ["" for _ in ids],
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote template: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
