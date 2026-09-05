from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


def _best_binary_threshold(feature: pd.Series, target: pd.Series) -> dict[str, Any] | None:
    valid = feature.notna() & target.notna()
    x = pd.to_numeric(feature[valid], errors="coerce")
    y = target[valid]
    valid_numeric = x.notna()
    x = x[valid_numeric]
    y = y[valid_numeric]
    if len(x) < 30 or y.nunique() != 2 or x.nunique() < 2:
        return None

    values = np.sort(x.unique())
    thresholds = (values[:-1] + values[1:]) / 2
    if len(thresholds) > 300:
        indexes = np.linspace(0, len(thresholds) - 1, 300).astype(int)
        thresholds = thresholds[indexes]

    labels = list(sorted(y.unique()))
    best = {"accuracy": 0.0, "threshold": None, "operator": None, "positive_label": labels[-1]}
    for threshold in thresholds:
        lower = (x <= threshold).astype(int)
        y_binary = (y == labels[-1]).astype(int)
        lower_accuracy = float((lower == y_binary).mean())
        upper_accuracy = float(((1 - lower) == y_binary).mean())
        if lower_accuracy > best["accuracy"]:
            best.update(accuracy=lower_accuracy, threshold=float(threshold), operator="<=")
        if upper_accuracy > best["accuracy"]:
            best.update(accuracy=upper_accuracy, threshold=float(threshold), operator=">")
    return best


def detect_target_leakage(df: pd.DataFrame, target: str) -> list[dict[str, Any]]:
    if target not in df.columns:
        raise KeyError(f"目标字段不存在: {target}")

    findings: list[dict[str, Any]] = []
    semantic_risks = {
        "is_congested": {"congestion_level", "speed_kmh"},
        "speed_kmh": {"is_congested", "congestion_level"},
    }
    for column in sorted(semantic_risks.get(target, set()) & set(df.columns)):
        findings.append({
            "column": column,
            "severity": "high",
            "type": "semantic",
            "message": f"{column} 与目标 {target} 存在直接业务定义或结果关系。",
        })

    if df[target].nunique(dropna=True) == 2:
        for column in df.select_dtypes(include="number").columns:
            if column == target:
                continue
            result = _best_binary_threshold(df[column], df[target])
            if not result or result["accuracy"] < 0.90:
                continue
            severity = "critical" if result["accuracy"] >= 0.98 else "high"
            findings.append({
                "column": column,
                "severity": severity,
                "type": "threshold_rule",
                "accuracy": round(result["accuracy"], 4),
                "threshold": round(float(result["threshold"]), 6),
                "operator": result["operator"],
                "message": (
                    f"单字段规则 {column} {result['operator']} {result['threshold']:.3f} "
                    f"可复现目标标签，匹配率 {result['accuracy']:.2%}。"
                ),
            })

    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for finding in findings:
        key = (finding["column"], finding["type"])
        deduplicated[key] = finding
    return list(deduplicated.values())


def detect_redundancy(df: pd.DataFrame) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if {"timestamp", "hour"}.issubset(df.columns):
        derived_hour = pd.to_datetime(df["timestamp"], errors="coerce").dt.hour
        valid = derived_hour.notna() & df["hour"].notna()
        if valid.any() and (derived_hour[valid] == df.loc[valid, "hour"]).all():
            findings.append({
                "type": "derived",
                "source": "timestamp",
                "dependent": "hour",
                "message": "hour 可由 timestamp 完整生成。",
            })

    candidates = [
        column
        for column in df.columns
        if 1 < df[column].nunique(dropna=True) <= 20 and not df[column].isna().any()
    ]
    seen: set[tuple[str, str]] = set()
    for left, right in combinations(candidates, 2):
        left_to_right = df.groupby(left, dropna=False)[right].nunique(dropna=False).max() == 1
        right_to_left = df.groupby(right, dropna=False)[left].nunique(dropna=False).max() == 1
        if left_to_right and right_to_left:
            key = tuple(sorted((left, right)))
            if key in seen:
                continue
            seen.add(key)
            findings.append({
                "type": "one_to_one_mapping",
                "source": left,
                "dependent": right,
                "message": f"{left} 与 {right} 在当前数据中一一对应，编码时应避免重复表达。",
            })
    return findings


def recommended_exclusions(df: pd.DataFrame, target: str) -> list[str]:
    exclusions = [target]
    for column in df.columns:
        lowered = column.lower()
        if lowered in {"record_id", "id", "index"} or lowered.endswith("_record_id"):
            exclusions.append(column)
    for finding in detect_target_leakage(df, target):
        if finding["severity"] in {"critical", "high"}:
            exclusions.append(finding["column"])
    # 保留更稳定、可解释的代表字段，减少同一信息被重复编码。
    if "road_id" in df.columns:
        exclusions.extend(column for column in ["road_name", "district"] if column in df.columns)
    if "dayofweek" in df.columns:
        exclusions.extend(column for column in ["timestamp", "hour"] if column in df.columns)
    return sorted(set(exclusions), key=lambda item: list(df.columns).index(item))
