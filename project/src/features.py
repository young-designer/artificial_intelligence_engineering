from __future__ import annotations

from typing import Any

import pandas as pd


def make_feature_frame(
    records: list[dict[str, Any]],
    feature_columns: list[str],
    boolean_columns: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")

    for column in boolean_columns:
        if column in frame.columns:
            frame[column] = frame[column].astype(bool).astype(int)

    return frame[feature_columns]
