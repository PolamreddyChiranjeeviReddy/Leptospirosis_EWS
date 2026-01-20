# Leptospirosis Ward-Level Early Warning System (EWS)

This folder contains **your original project implementation** built by referencing the provided repos:
- Argentina EWS (two-stage / rolling prediction + outbreak detection): `../lepto-argentina-main/`
- Fiji climate lagged models (weekly/monthly lags): `../leptospirosis_Fiji_2023-main/`
- Kelantan RF input structure (optimized hydro-meteorological indices): `../Kelantan_leptospirosis_modelling-main/`
- Meteorology/runoff lag studies: `../meteorology_leptospirosis-main/`

Goal: predict ward/village **risk probability** 1–3 weeks ahead using climate + flood + sanitation signals, then produce **risk categories** and **GIS maps**, with **SHAP** explanations.

## What you get
- Data alignment to **weekly** ward-week rows
- Feature engineering with **1–3 week lags** (Argentina/Fiji-style)
- **Rolling (time-aware) validation** (avoids leakage)
- **XGBoost** classifier (primary model)
- **SHAP** explanations (global + per-ward)
- **Risk thresholding** (low/medium/high + consecutive-week rule)
- Ward-level **GeoJSON + Folium HTML** risk maps (including weekly time slider)

## Quickstart (Windows)

### 1) Create a Python environment
From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Optional (only if you want the LSTM model):

```powershell
pip install -r requirements-optional.txt
```

### 2) Run the smoke test (no external data)
This generates a tiny synthetic ward boundary + weekly data and runs the full pipeline.

```powershell
python .\scripts\smoke_test.py
```

Outputs go to `outputs/`.

## Bring your real data

### Required inputs (minimal)
You can point the pipeline at your own files, but the expected *schemas* are:

1) **Ward boundaries** (GeoPackage/Shapefile)
- Must include a ward id column, e.g. `ward_id`

2) **Cases** (CSV)
- Columns: `ward_id`, `date`, `cases`
- `date` can be daily or weekly; it will be aggregated to weekly.

3) **Climate** (CSV)
- Columns: `ward_id`, `date`, `rain_mm`, `tmean_c` (optional), `rh_pct` (optional)

4) **Sanitation** (CSV, static)
- Columns: `ward_id`, `open_drain_pct`, `no_toilet_pct`, `unsafe_water_pct`, `pop_density` (optional)

5) **Flood** (choose one)
- **Option A (simplest):** weekly ward-level CSV with `ward_id`, `week_start`, `flooded_area_pct`, `flood_presence`
- **Option B:** GeoTIFF rasters per week (from GEE/Sentinel-1), plus ward boundaries for zonal stats.

### Run on real data
Create a config file (example below) and run:

```powershell
python -m lepto_ews.cli run --config .\config.yaml
```

Example `config.yaml`:

```yaml
boundaries_path: data/real/wards.gpkg
boundaries_id_col: ward_id

cases_csv: data/real/cases.csv
climate_csv: data/real/climate.csv
sanitation_csv: data/real/sanitation.csv

# flood inputs (choose ONE)
flood_weekly_csv: data/real/flood_weekly.csv
# flood_raster_dir: data/real/flood_rasters

prediction_horizon_weeks: 1
lags_weeks: [1, 2, 3]

labeling:
  method: quantile
  quantile: 0.75

model:
  xgb:
    n_estimators: 500
    max_depth: 5
    learning_rate: 0.05
    subsample: 0.9
    colsample_bytree: 0.9

thresholds:
  low: 0.4
  high: 0.7
  consecutive_weeks: 2
```

## Notes (how this aligns with your references)
- **Lags + climate-driven signal**: Fiji & Argentina repos use lagged climatic indicators; this project uses the same principle but at ward-week level.
- **Rolling evaluation**: Argentina’s out-of-sample forecasting approach is matched via time-based rolling splits.
- **RF vs XGBoost**: Kelantan repo uses RF with optimized indices; this project extends to XGBoost and keeps inputs compatible.
- **Thresholding**: Inspired by CDC-style logic, but applied to **predicted risk** (probability) + consecutive-week rule.

## Folder layout
- `src/lepto_ews/`: pipeline implementation
- `scripts/`: runnable scripts (smoke test)
- `data/`: sample + your real data
- `outputs/`: model outputs, SHAP artifacts, maps
