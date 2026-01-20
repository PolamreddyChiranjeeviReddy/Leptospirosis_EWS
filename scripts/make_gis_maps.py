from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lepto_ews.gis.mapping import make_latest_risk_map, make_timeslider_map
from lepto_ews.io import read_boundaries
from lepto_ews.run_metadata import finalize_run_metadata


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate GIS HTML maps for an existing run directory")
    ap.add_argument("--run-dir", type=Path, required=True, help="Run directory (e.g., outputs/fiji_run_h3)")
    args = ap.parse_args()

    run_dir = args.run_dir
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing run metadata: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg = meta.get("config", {})

    boundaries_path = cfg.get("boundaries_path")
    id_col = cfg.get("boundaries_id_col")
    if not boundaries_path or not id_col:
        raise ValueError("run_metadata.json config is missing boundaries_path/boundaries_id_col")

    boundaries = read_boundaries(Path(boundaries_path), str(id_col))

    pred_path = run_dir / "predictions_thresholded.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions: {pred_path}")

    pred_df = pd.read_parquet(pred_path)

    make_latest_risk_map(boundaries, pred_df, str(id_col), run_dir / "risk_latest_map.html")
    make_timeslider_map(boundaries, pred_df, str(id_col), run_dir / "risk_timeslider_map.html")

    # Keep provenance consistent when we update HTML outputs.
    finalize_run_metadata(run_dir)


if __name__ == "__main__":
    main()
