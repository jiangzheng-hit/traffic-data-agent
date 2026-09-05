import json

import pandas as pd

from traffic_data_agent.agent import OllamaPlanner


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({"response": json.dumps(self.body, ensure_ascii=False)}).encode("utf-8")


def test_ollama_cannot_remove_safety_steps_or_leakage_exclusions(monkeypatch):
    df = pd.DataFrame({
        "record_id": [f"T{i}" for i in range(40)],
        "timestamp": pd.date_range("2026-01-01", periods=40, freq="h").astype(str),
        "speed_kmh": range(40),
        "congestion_level": ["severe" if value < 20 else "free" for value in range(40)],
        "is_congested": [1 if value < 20 else 0 for value in range(40)],
    })
    unsafe = {
        "goal": "绕过安全限制",
        "split_strategy": "random",
        "steps": [
            {"tool": "exclude_features", "reason": "只排除目标", "parameters": {"columns": ["is_congested"]}},
            {"tool": "train_baseline", "reason": "训练", "parameters": {}},
        ],
        "warnings": [],
    }
    monkeypatch.setattr("traffic_data_agent.agent.request.urlopen", lambda *args, **kwargs: FakeResponse(unsafe))
    plan = OllamaPlanner().plan("预测拥堵", df, "is_congested")
    tools = [step.tool for step in plan.steps]
    assert "validate_dataset" in tools
    assert "generate_report" in tools
    exclusions = next(step.parameters["columns"] for step in plan.steps if step.tool == "exclude_features")
    assert "speed_kmh" in exclusions
    assert "congestion_level" in exclusions
