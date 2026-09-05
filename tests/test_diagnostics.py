import pandas as pd

from traffic_data_agent.diagnostics import detect_redundancy, detect_target_leakage, recommended_exclusions
from traffic_data_agent.profiling import calculate_quality_score, profile_dataset


def test_detects_exact_threshold_leakage():
    df = pd.DataFrame({
        "speed_kmh": list(range(10, 50)),
        "is_congested": [1 if value < 35 else 0 for value in range(10, 50)],
    })
    findings = detect_target_leakage(df, "is_congested")
    threshold = [item for item in findings if item["type"] == "threshold_rule"]
    assert threshold
    assert threshold[0]["accuracy"] == 1.0


def test_detects_derived_hour():
    df = pd.DataFrame({
        "timestamp": ["2026-05-01 07:00", "2026-05-01 08:00"],
        "hour": [7, 8],
    })
    findings = detect_redundancy(df)
    assert any(item["type"] == "derived" for item in findings)


def test_recommends_leakage_exclusions():
    values = list(range(40))
    df = pd.DataFrame({
        "record_id": [f"T{i}" for i in values],
        "speed_kmh": values,
        "congestion_level": ["severe" if value < 20 else "free" for value in values],
        "is_congested": [1 if value < 20 else 0 for value in values],
    })
    exclusions = recommended_exclusions(df, "is_congested")
    assert {"record_id", "speed_kmh", "congestion_level", "is_congested"}.issubset(exclusions)


def test_quality_score_penalizes_critical_leakage():
    df = pd.DataFrame({"speed": range(40), "target": [1 if x < 20 else 0 for x in range(40)]})
    leakage = detect_target_leakage(df, "target")
    score = calculate_quality_score(profile_dataset(df, "target"), leakage, [])
    assert score["score"] <= 65
    assert score["penalties"]["critical_leakage"] == 35.0
