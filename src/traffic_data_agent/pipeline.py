from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .cleaning import create_time_features, impute_missing, remove_duplicates
from .diagnostics import detect_redundancy, detect_target_leakage, recommended_exclusions
from .experiments import run_comparison_experiments
from .modeling import train_model_suite
from .profiling import build_data_dictionary, calculate_quality_score, profile_dataset
from .reporting import build_html_report, build_markdown_report
from .schemas import ExecutionPlan, WorkflowResult
from .validation import validate_dataset


class TrafficDataWorkflow:
    def __init__(self, df: pd.DataFrame):
        self.original = df.copy()

    def execute(self, plan: ExecutionPlan) -> WorkflowResult:
        plan.validate()
        target = plan.target
        if target not in self.original.columns:
            raise KeyError(f"目标字段不存在: {target}")

        profile_before = profile_dataset(self.original, target)
        leakage = detect_target_leakage(self.original, target)
        redundancy = detect_redundancy(self.original)
        quality = calculate_quality_score(profile_before, leakage, redundancy)
        working = self.original.copy()
        logs = []

        def trace(action: str, detail: str, affected: int = 0) -> None:
            logs.append({
                "time": datetime.now().isoformat(timespec="seconds"),
                "action": action,
                "detail": detail,
                "affected": int(affected),
            })

        for step in plan.steps:
            if step.tool == "remove_duplicates":
                working, log = remove_duplicates(working)
                logs.append(log)
            elif step.tool == "impute_missing":
                working, log = impute_missing(working)
                logs.append(log)
            elif step.tool == "create_time_features":
                working, log = create_time_features(working)
                logs.append(log)

        exclusions = recommended_exclusions(working, target)
        for step in plan.steps:
            if step.tool == "exclude_features":
                requested = step.parameters.get("columns", [])
                exclusions = list(dict.fromkeys([*exclusions, *requested]))
                trace("exclude_features", f"建模阶段排除字段: {exclusions}", len(exclusions))

        validation = validate_dataset(working)
        trace("validate_dataset", "所有数据质量检查通过" if validation["passed"] else "数据质量检查未通过", len(validation["checks"]))
        if not validation["passed"]:
            raise ValueError("数据质量验证未通过，停止建模")

        metrics = train_model_suite(working, target, exclusions, plan.split_strategy)
        trace("train_baseline", f"完成 {metrics['model']}，采用 {plan.split_strategy} 划分", metrics["test_rows"])
        experiments = run_comparison_experiments(working, target)
        trace("compare_experiments", "完成泄漏字段与数据划分对照实验", len(experiments))
        profile_after = profile_dataset(working, target)
        trace("generate_report", "生成Markdown、HTML、图表和结构化结果", 1)
        return WorkflowResult(
            cleaned_data=working,
            profile_before=profile_before,
            profile_after=profile_after,
            leakage_findings=leakage,
            redundancy_findings=redundancy,
            validation=validation,
            metrics=metrics,
            experiments=experiments,
            quality=quality,
            logs=logs,
        )

    @staticmethod
    def save_outputs(result: WorkflowResult, plan: ExecutionPlan, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        result.cleaned_data.to_csv(output_dir / "cleaned_traffic_data.csv", index=False, encoding="utf-8-sig")
        exclusions = result.metrics.get("excluded_features", [])
        build_data_dictionary(result.cleaned_data, plan.target, exclusions).to_csv(
            output_dir / "data_dictionary.csv", index=False, encoding="utf-8-sig"
        )
        (output_dir / "execution_plan.json").write_text(
            json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "processing_log.json").write_text(
            json.dumps(result.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "model_metrics.json").write_text(
            json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "comparison_experiments.json").write_text(
            json.dumps(result.experiments, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report = build_markdown_report(
            result.profile_before,
            result.profile_after,
            result.leakage_findings,
            result.redundancy_findings,
            result.validation,
            result.metrics,
            result.experiments,
            result.quality,
            result.logs,
        )
        (output_dir / "analysis_report.md").write_text(report, encoding="utf-8")
        from .visualization import save_figures

        save_figures(result.cleaned_data, result.experiments, output_dir / "figures")
        (output_dir / "analysis_report.html").write_text(
            build_html_report(report, result.quality, result.experiments, result.metrics), encoding="utf-8"
        )
