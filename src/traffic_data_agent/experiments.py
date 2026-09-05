from __future__ import annotations

from typing import Any

import pandas as pd

from .diagnostics import recommended_exclusions
from .modeling import train_baseline


def _id_exclusions(df: pd.DataFrame, target: str) -> list[str]:
    result = [target]
    for column in df.columns:
        lowered = column.lower()
        if lowered in {"record_id", "id", "index"} or lowered.endswith("_record_id"):
            result.append(column)
    return list(dict.fromkeys(result))


def run_comparison_experiments(df: pd.DataFrame, target: str) -> list[dict[str, Any]]:
    safe = recommended_exclusions(df, target)
    leaky = _id_exclusions(df, target)
    specifications = [
        ("leaky_random", "含泄漏字段 + 随机划分", leaky, "random", False),
        ("safe_random", "排除泄漏字段 + 随机划分", safe, "random", True),
        ("safe_time", "排除泄漏字段 + 时间划分", safe, "time", True),
    ]
    results: list[dict[str, Any]] = []
    for experiment_id, label, excluded, split, trusted in specifications:
        comparison_model = "decision_tree" if df[target].nunique(dropna=True) == 2 else "default"
        metrics = train_baseline(df, target, excluded, split, model_name=comparison_model)
        results.append({
            "id": experiment_id,
            "label": label,
            "trusted": trusted,
            "interpretation": (
                "演示目标泄漏如何制造虚高指标，不应用于模型结论。"
                if not trusted
                else "排除已识别的泄漏字段，可作为更可信的基线。"
            ),
            **metrics,
        })
    return results
