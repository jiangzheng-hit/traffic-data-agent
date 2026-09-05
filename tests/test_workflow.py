from pathlib import Path

import pandas as pd

from traffic_data_agent.agent import RulePlanner
from traffic_data_agent.experiments import run_comparison_experiments
from traffic_data_agent.pipeline import TrafficDataWorkflow


ROOT = Path(__file__).resolve().parents[1]


def test_course_dataset_workflow_runs():
    dataset = ROOT / "data" / "raw" / "traffic_ml_homework_dataset.csv"
    if not dataset.exists():
        return
    df = pd.read_csv(dataset)
    plan = RulePlanner().plan("清洗并预测拥堵", df, "is_congested", "time")
    result = TrafficDataWorkflow(df).execute(plan)
    assert result.profile_before["rows"] == 726
    assert result.profile_after["rows"] == 720
    assert result.profile_after["missing_cells"] == 0
    assert result.validation["passed"]
    assert result.metrics["task"] == "classification"
    assert result.metrics["top_features"]
    assert len(result.metrics["validation_comparison"]) == 3
    assert result.metrics["train_rows"] + result.metrics["validation_rows"] + result.metrics["test_rows"] == 720
    assert len(result.experiments) == 3
    assert result.experiments[0]["accuracy"] == 1.0
    assert any(log["action"] == "validate_dataset" for log in result.logs)


def test_output_bundle_contains_auditable_artifacts(tmp_path):
    dataset = ROOT / "data" / "raw" / "traffic_ml_homework_dataset.csv"
    df = pd.read_csv(dataset)
    plan = RulePlanner().plan("清洗并预测拥堵", df, "is_congested", "time")
    result = TrafficDataWorkflow(df).execute(plan)
    TrafficDataWorkflow.save_outputs(result, plan, tmp_path)
    expected = {
        "cleaned_traffic_data.csv",
        "data_dictionary.csv",
        "execution_plan.json",
        "processing_log.json",
        "model_metrics.json",
        "comparison_experiments.json",
        "analysis_report.md",
        "analysis_report.html",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    assert (tmp_path / "figures" / "experiments.png").exists()


def test_rule_plan_keeps_validation_before_modeling():
    dataset = ROOT / "data" / "raw" / "traffic_ml_homework_dataset.csv"
    df = pd.read_csv(dataset)
    plan = RulePlanner().plan("预测拥堵", df, "is_congested", "time")
    tools = [step.tool for step in plan.steps]
    assert tools.index("validate_dataset") < tools.index("train_baseline")
    excluded = next(step.parameters["columns"] for step in plan.steps if step.tool == "exclude_features")
    assert "speed_kmh" in excluded


def test_comparison_has_untrusted_leakage_demo():
    dataset = ROOT / "data" / "raw" / "traffic_ml_homework_dataset.csv"
    df = pd.read_csv(dataset).drop_duplicates()
    experiments = run_comparison_experiments(df, "is_congested")
    assert experiments[0]["trusted"] is False
    assert experiments[0]["accuracy"] == 1.0
    assert all(item["trusted"] for item in experiments[1:])
