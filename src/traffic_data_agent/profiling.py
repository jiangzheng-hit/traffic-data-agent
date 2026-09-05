from __future__ import annotations

from typing import Any

import pandas as pd


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def profile_dataset(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    missing = df.isna().sum()
    profile: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(missing.sum()),
        "missing_by_column": {k: int(v) for k, v in missing[missing > 0].items()},
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "unique_values": {column: int(df[column].nunique(dropna=True)) for column in df.columns},
    }

    if "timestamp" in df.columns:
        timestamp = pd.to_datetime(df["timestamp"], errors="coerce")
        profile["timestamp_invalid"] = int(timestamp.isna().sum())
        profile["timestamp_min"] = None if timestamp.dropna().empty else str(timestamp.min())
        profile["timestamp_max"] = None if timestamp.dropna().empty else str(timestamp.max())

    if target and target in df.columns:
        counts = df[target].value_counts(dropna=False)
        profile["target"] = target
        profile["target_distribution"] = {
            str(_json_value(label)): int(count) for label, count in counts.items()
        }

    return profile


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().transpose().reset_index(names="field")


def calculate_quality_score(
    profile: dict[str, Any],
    leakage: list[dict[str, Any]],
    redundancy: list[dict[str, Any]],
) -> dict[str, Any]:
    total_cells = max(1, profile["rows"] * profile["columns"])
    duplicate_rate = profile["duplicate_rows"] / max(1, profile["rows"])
    missing_rate = profile["missing_cells"] / total_cells
    critical_count = sum(item.get("severity") == "critical" for item in leakage)
    high_count = sum(item.get("severity") == "high" for item in leakage)

    penalties = {
        "duplicates": min(12.0, duplicate_rate * 500),
        "missing": min(12.0, missing_rate * 1000),
        "critical_leakage": min(35.0, critical_count * 35.0),
        "high_leakage": min(18.0, high_count * 9.0),
        "redundancy": min(13.0, len(redundancy) * 1.8),
    }
    score = max(0, round(100 - sum(penalties.values())))
    if score >= 85:
        level = "良好"
    elif score >= 70:
        level = "一般"
    elif score >= 50:
        level = "较高风险"
    else:
        level = "高风险"
    return {
        "score": score,
        "level": level,
        "penalties": {key: round(value, 1) for key, value in penalties.items()},
        "explanation": "该评分用于排列数据治理优先级，不代表数据在所有业务场景中的绝对质量。",
    }


def build_data_dictionary(
    df: pd.DataFrame,
    target: str,
    excluded: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        lowered = column.lower()
        if column == target:
            role = "target"
            recommendation = "仅作为预测目标，不进入特征"
        elif column in excluded:
            role = "excluded"
            recommendation = "建模阶段排除，原因见泄漏风险或标识字段诊断"
        elif "time" in lowered or lowered in {"hour", "dayofweek", "is_peak_hour"}:
            role = "time"
            recommendation = "可用于时间特征或时间顺序划分"
        elif pd.api.types.is_numeric_dtype(df[column]):
            role = "numeric_feature"
            recommendation = "可用，建模Pipeline中进行缺失值处理和标准化"
        else:
            role = "categorical_feature"
            recommendation = "可用，建模Pipeline中进行众数填补和One-Hot编码"
        sample_values = [str(value) for value in df[column].dropna().unique()[:3]]
        rows.append({
            "field": column,
            "dtype": str(df[column].dtype),
            "role": role,
            "non_null": int(df[column].notna().sum()),
            "missing": int(df[column].isna().sum()),
            "unique": int(df[column].nunique(dropna=True)),
            "samples": " | ".join(sample_values),
            "recommendation": recommendation,
        })
    return pd.DataFrame(rows)
