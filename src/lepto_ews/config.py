from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class LabelingConfig(BaseModel):
    method: Literal["quantile", "mean_plus_kstd"] = "quantile"
    quantile: float = 0.75
    k_std: float = 1.0
    time_aware: bool = True
    window_weeks: int = 52
    min_periods: int = 12


class RainAnomalyConfig(BaseModel):
    time_aware: bool = True
    window_weeks: int = 52
    min_periods: int = 12


class FeatureEngineeringConfig(BaseModel):
    rain_anomaly: RainAnomalyConfig = Field(default_factory=RainAnomalyConfig)


class XGBConfig(BaseModel):
    n_estimators: int = 500
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0
    tree_method: Literal["auto", "hist"] = "hist"
    random_state: int = 42


class ModelConfig(BaseModel):
    xgb: XGBConfig = Field(default_factory=XGBConfig)


class TuningConfig(BaseModel):
    enabled: bool = True
    n_trials: int = 20
    validation_weeks: int = 8
    metric: Literal["pr_auc", "auc"] = "pr_auc"
    random_state: int = 42


class CalibrationConfig(BaseModel):
    method: Literal["none", "sigmoid", "isotonic"] = "sigmoid"


class TrainingConfig(BaseModel):
    early_stopping_rounds: int = 50
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)


class ThresholdConfig(BaseModel):
    low: float = 0.4
    high: float = 0.7
    consecutive_weeks: int = 2


class AppConfig(BaseModel):
    boundaries_path: Path | None = None
    boundaries_id_col: str = "ward_id"

    cases_csv: Path
    climate_csv: Path | None = None
    sanitation_csv: Path | None = None

    flood_weekly_csv: Path | None = None
    flood_raster_dir: Path | None = None

    prediction_horizon_weeks: int = 1
    lags_weeks: list[int] = Field(default_factory=lambda: [1, 2, 3])

    features: FeatureEngineeringConfig = Field(default_factory=FeatureEngineeringConfig)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)

    output_dir: Path = Path("outputs")

    def validate_sources(self) -> None:
        # Flood data is optional. If provided, ensure only one source is set.
        if self.flood_weekly_csv is not None and self.flood_raster_dir is not None:
            raise ValueError("Provide only one of flood_weekly_csv or flood_raster_dir")
