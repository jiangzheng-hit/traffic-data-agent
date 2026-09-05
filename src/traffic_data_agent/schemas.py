from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ALLOWED_TOOLS = {
    "remove_duplicates",
    "impute_missing",
    "create_time_features",
    "exclude_features",
    "validate_dataset",
    "train_baseline",
    "generate_report",
}


@dataclass
class PlanStep:
    tool: str
    reason: str
    parameters: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False

    def validate(self) -> None:
        if self.tool not in ALLOWED_TOOLS:
            raise ValueError(f"不允许调用工具: {self.tool}")
        if not isinstance(self.parameters, dict):
            raise ValueError("parameters 必须是字典")


@dataclass
class ExecutionPlan:
    goal: str
    target: str
    split_strategy: str
    steps: list[PlanStep]
    warnings: list[str] = field(default_factory=list)
    planner: str = "rule"
    fallback_reason: str | None = None

    def validate(self) -> None:
        if self.split_strategy not in {"random", "time"}:
            raise ValueError("split_strategy 只能是 random 或 time")
        if not self.steps:
            raise ValueError("执行计划不能为空")
        for step in self.steps:
            step.validate()
        tools = [step.tool for step in self.steps]
        required = {"validate_dataset", "train_baseline", "generate_report"}
        missing = required - set(tools)
        if missing:
            raise ValueError(f"执行计划缺少必要步骤: {sorted(missing)}")
        if tools.index("validate_dataset") > tools.index("train_baseline"):
            raise ValueError("必须先验证数据，再进行建模")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowResult:
    cleaned_data: Any
    profile_before: dict[str, Any]
    profile_after: dict[str, Any]
    leakage_findings: list[dict[str, Any]]
    redundancy_findings: list[dict[str, Any]]
    validation: dict[str, Any]
    metrics: dict[str, Any]
    experiments: list[dict[str, Any]]
    quality: dict[str, Any]
    logs: list[dict[str, Any]]
