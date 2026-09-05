from __future__ import annotations

from typing import Any

import pandas as pd


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    duplicates = int(df.duplicated().sum())
    missing = int(df.isna().sum().sum())
    add("no_duplicates", duplicates == 0, f"重复记录 {duplicates} 条")
    add("no_missing", missing == 0, f"缺失单元格 {missing} 个")

    if "timestamp" in df.columns:
        invalid = int(pd.to_datetime(df["timestamp"], errors="coerce").isna().sum())
        add("valid_timestamp", invalid == 0, f"无效时间 {invalid} 条")
    if "occupancy" in df.columns:
        invalid = int((~df["occupancy"].between(0, 1)).sum())
        add("occupancy_range", invalid == 0, f"超出[0, 1]范围 {invalid} 条")
    if "speed_kmh" in df.columns:
        invalid = int((df["speed_kmh"] < 0).sum())
        add("speed_non_negative", invalid == 0, f"负速度 {invalid} 条")
    if "volume" in df.columns:
        invalid = int((df["volume"] < 0).sum())
        add("volume_non_negative", invalid == 0, f"负流量 {invalid} 条")

    return {"passed": all(check["passed"] for check in checks), "checks": checks}

