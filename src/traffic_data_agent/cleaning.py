from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def _log(action: str, detail: str, affected: int = 0) -> dict[str, Any]:
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "action": action,
        "detail": detail,
        "affected": int(affected),
    }


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    count = int(df.duplicated().sum())
    return df.drop_duplicates().copy(), _log("remove_duplicates", "删除完全重复记录", count)


def impute_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = df.copy()
    before = int(output.isna().sum().sum())
    strategies: dict[str, str] = {}
    for column in output.columns:
        if not output[column].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(output[column]):
            value = output[column].median()
            strategies[column] = f"median={value}"
        else:
            modes = output[column].mode(dropna=True)
            value = modes.iloc[0] if not modes.empty else "UNKNOWN"
            strategies[column] = f"mode={value}"
        output[column] = output[column].fillna(value)
    return output, _log("impute_missing", f"填补策略: {strategies}", before)


def create_time_features(
    df: pd.DataFrame,
    morning: tuple[int, int] = (7, 9),
    evening: tuple[int, int] = (17, 19),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "timestamp" not in df.columns:
        return df.copy(), _log("create_time_features", "未找到 timestamp，跳过", 0)
    output = df.copy()
    timestamp = pd.to_datetime(output["timestamp"], errors="coerce")
    output["dayofweek"] = timestamp.dt.dayofweek
    hour = timestamp.dt.hour
    output["is_peak_hour"] = (
        hour.between(morning[0], morning[1]) | hour.between(evening[0], evening[1])
    ).astype("int64")
    return output, _log("create_time_features", "新增 dayofweek 与 is_peak_hour", len(output))


def exclude_features(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    existing = [column for column in columns if column in df.columns]
    return df.drop(columns=existing).copy(), _log("exclude_features", f"排除字段: {existing}", len(existing))

