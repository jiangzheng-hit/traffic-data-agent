from __future__ import annotations

import json
from urllib import request

import pandas as pd

from .diagnostics import detect_target_leakage, recommended_exclusions
from .schemas import ExecutionPlan, PlanStep


class RulePlanner:
    def plan(self, prompt: str, df: pd.DataFrame, target: str, split_strategy: str = "time") -> ExecutionPlan:
        duplicates = int(df.duplicated().sum())
        missing = int(df.isna().sum().sum())
        exclusions = recommended_exclusions(df, target)
        leakage = detect_target_leakage(df, target)
        warnings = [finding["message"] for finding in leakage]

        steps: list[PlanStep] = []
        if duplicates:
            steps.append(PlanStep("remove_duplicates", f"发现 {duplicates} 条重复记录", requires_confirmation=True))
        if missing:
            steps.append(PlanStep("impute_missing", f"发现 {missing} 个缺失单元格", requires_confirmation=True))
        if "timestamp" in df.columns:
            steps.append(PlanStep("create_time_features", "从时间字段构造星期与高峰特征"))
        steps.append(PlanStep(
            "exclude_features",
            "排除目标、标识和高风险泄漏字段",
            {"columns": exclusions},
            requires_confirmation=True,
        ))
        steps.extend([
            PlanStep("validate_dataset", "建模前执行数据质量检查"),
            PlanStep("train_baseline", "使用处理后的安全特征建立基线模型"),
            PlanStep("generate_report", "汇总数据、风险、日志与模型指标"),
        ])
        plan = ExecutionPlan(
            goal=prompt.strip() or f"处理数据并预测 {target}",
            target=target,
            split_strategy=split_strategy,
            steps=steps,
            warnings=warnings,
            planner="rule",
        )
        plan.validate()
        return plan


class OllamaPlanner:
    def __init__(self, model: str = "qwen2.5:3b", endpoint: str = "http://127.0.0.1:11434/api/generate"):
        self.model = model
        self.endpoint = endpoint

    def plan(self, prompt: str, df: pd.DataFrame, target: str, split_strategy: str = "time") -> ExecutionPlan:
        fallback = RulePlanner().plan(prompt, df, target, split_strategy)
        system_prompt = f"""
你是本地交通数据处理规划器。只能输出JSON，不得输出代码或解释。
允许工具：remove_duplicates, impute_missing, create_time_features, exclude_features,
validate_dataset, train_baseline, generate_report。
目标字段：{target}
字段：{list(df.columns)}
基础安全计划：{json.dumps(fallback.to_dict(), ensure_ascii=False)}
用户要求：{prompt}
请返回与基础计划同结构的JSON。不得增加白名单外工具，不得删除验证步骤。
""".strip()
        payload = json.dumps({"model": self.model, "prompt": system_prompt, "stream": False}).encode("utf-8")
        req = request.Request(self.endpoint, data=payload, headers={"Content-Type": "application/json"})
        with request.urlopen(req, timeout=60) as response:
            answer = json.loads(response.read().decode("utf-8"))["response"]
        answer = answer.strip()
        if answer.startswith("```"):
            answer = answer.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        raw = json.loads(answer)
        proposed_steps = [PlanStep(**step) for step in raw["steps"]]
        proposed_by_tool = {step.tool: step for step in proposed_steps}

        # LLM可以改善理由和非安全参数，但不能删除必需步骤或解除字段排除。
        steps: list[PlanStep] = []
        for safe_step in fallback.steps:
            proposed = proposed_by_tool.get(safe_step.tool)
            if not proposed:
                steps.append(safe_step)
                continue
            parameters = proposed.parameters
            if safe_step.tool == "exclude_features":
                safe_columns = set(safe_step.parameters.get("columns", []))
                proposed_columns = set(parameters.get("columns", []))
                parameters = {"columns": sorted(safe_columns | proposed_columns)}
            steps.append(PlanStep(
                tool=safe_step.tool,
                reason=proposed.reason or safe_step.reason,
                parameters=parameters,
                requires_confirmation=safe_step.requires_confirmation or proposed.requires_confirmation,
            ))
        plan = ExecutionPlan(
            goal=str(raw.get("goal", fallback.goal)),
            target=target,
            split_strategy=str(raw.get("split_strategy", split_strategy)),
            steps=steps,
            warnings=list(raw.get("warnings", fallback.warnings)),
            planner="ollama",
        )
        plan.validate()
        required = {"validate_dataset", "train_baseline", "generate_report"}
        if not required.issubset({step.tool for step in plan.steps}):
            raise ValueError("Ollama计划缺少必要的验证、建模或报告步骤")
        return plan


class HybridPlanner:
    """Try Ollama locally and fall back to deterministic planning without blocking the workflow."""

    def __init__(self, model: str = "qwen2.5:3b"):
        self.model = model

    def plan(self, prompt: str, df: pd.DataFrame, target: str, split_strategy: str = "time") -> ExecutionPlan:
        try:
            return OllamaPlanner(self.model).plan(prompt, df, target, split_strategy)
        except Exception as exc:
            plan = RulePlanner().plan(prompt, df, target, split_strategy)
            plan.planner = "rule_fallback"
            plan.fallback_reason = f"Ollama不可用或计划未通过校验: {type(exc).__name__}"
            plan.warnings.append(plan.fallback_reason)
            return plan
