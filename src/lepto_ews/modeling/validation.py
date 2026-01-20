from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RollingSplit:
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def rolling_time_splits(weeks: list[pd.Timestamp], horizon_weeks: int, min_train_weeks: int = 26) -> list[RollingSplit]:
    """Create expanding-window splits on a sorted list of week_start timestamps."""
    weeks = sorted(pd.to_datetime(weeks))
    splits: list[RollingSplit] = []

    for i in range(min_train_weeks, len(weeks) - horizon_weeks):
        train_end = weeks[i - 1]
        test_start = weeks[i]
        test_end = weeks[i + horizon_weeks - 1]
        splits.append(RollingSplit(train_end=train_end, test_start=test_start, test_end=test_end))

    return splits
