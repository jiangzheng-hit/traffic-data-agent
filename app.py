from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from traffic_data_agent.agent import HybridPlanner, RulePlanner
from traffic_data_agent.diagnostics import detect_redundancy, detect_target_leakage
from traffic_data_agent.pipeline import TrafficDataWorkflow
from traffic_data_agent.diagnostics import recommended_exclusions
from traffic_data_agent.profiling import build_data_dictionary, calculate_quality_score, profile_dataset
from traffic_data_agent.visualization import (
    confusion_figure,
    experiment_figure,
    missing_figure,
    traffic_relationship_figure,
)


st.set_page_config(page_title="交通数据质量审查与建模助手", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1240px; padding-top: 1.6rem; padding-bottom: 3rem;}
[data-testid="stMetric"] {background:#f7fafb;border:1px solid #dce7ea;border-radius:12px;padding:14px 16px;}
[data-testid="stMetricValue"] {color:#17324d;}
.hero {padding:24px 28px;border-radius:16px;background:linear-gradient(120deg,#17324d,#0d7c86);color:white;margin-bottom:18px;}
.hero h1 {margin:0 0 7px 0;font-size:2rem;color:white;}.hero p{margin:0;color:#e4f4f5;}
.risk-critical {padding:14px 16px;border-left:5px solid #c94c4c;background:#fff3f3;border-radius:7px;margin:8px 0 16px 0;}
.safe-note {padding:14px 16px;border-left:5px solid #0d7c86;background:#eff9f8;border-radius:7px;margin:8px 0;}
</style>
<div class="hero"><h1>交通数据质量审查与建模助手</h1>
<p>Agent 制定受约束的执行计划，确定性 Python 工具完成计算，验证未通过则停止建模。</p></div>
""", unsafe_allow_html=True)

default_file = ROOT / "data" / "raw" / "traffic_ml_homework_dataset.csv"
uploaded = st.sidebar.file_uploader("选择交通数据 CSV", type=["csv"])
planner_mode = st.sidebar.selectbox("规划模式", ["规则规划器", "Ollama本地模型"])
ollama_model = st.sidebar.text_input("Ollama模型", "qwen2.5:3b")

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif default_file.exists():
    df = pd.read_csv(default_file)
    st.sidebar.info("当前使用课程样例数据")
else:
    st.warning("请上传 CSV 文件。")
    st.stop()

numeric_targets = [
    column for column in df.select_dtypes(include="number").columns
    if df[column].nunique(dropna=True) >= 2
]
targets = [column for column in ["is_congested", "speed_kmh"] if column in numeric_targets]
targets += [column for column in numeric_targets if column not in targets]
if not targets:
    st.error("当前数据没有可用于分类或回归的数值目标字段。")
    st.stop()
target = st.sidebar.selectbox("预测目标", targets)
split_strategy = st.sidebar.selectbox("数据划分", ["time", "random"], format_func=lambda x: "时间顺序" if x == "time" else "随机分层")

data_signature = f"{df.shape}|{tuple(df.columns)}|{int(pd.util.hash_pandas_object(df, index=True).sum())}"
session_signature = (data_signature, target, split_strategy)
if st.session_state.get("session_signature") != session_signature:
    st.session_state["session_signature"] = session_signature
    st.session_state.pop("plan", None)
    st.session_state.pop("result", None)

profile = profile_dataset(df, target)
initial_leakage = detect_target_leakage(df, target)
initial_redundancy = detect_redundancy(df)
quality = calculate_quality_score(profile, initial_leakage, initial_redundancy)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("数据行数", profile["rows"])
col2.metric("字段数量", profile["columns"])
col3.metric("重复记录", profile["duplicate_rows"])
col4.metric("缺失单元格", profile["missing_cells"])
col5.metric("治理前评分", f"{quality['score']}/100")
col5.caption(f"风险等级：{quality['level']}")

tab1, tab2, tab3, tab4 = st.tabs(["数据概览", "风险诊断", "Agent计划", "执行结果"])
with tab1:
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("数据预览")
        st.dataframe(df.head(30), width="stretch", height=385)
    with right:
        st.subheader("缺失值分布")
        st.pyplot(missing_figure(df), width="stretch")
    with st.expander("查看字段字典与建模建议"):
        dictionary = build_data_dictionary(df, target, recommended_exclusions(df, target))
        st.dataframe(dictionary, hide_index=True, width="stretch")
    st.subheader("交通运行关系")
    st.pyplot(traffic_relationship_figure(df), width="stretch")

with tab2:
    leakage = initial_leakage
    redundancy = initial_redundancy
    critical = [item for item in leakage if item.get("severity") == "critical"]
    if critical:
        st.markdown(
            f"<div class='risk-critical'><b>发现严重目标泄漏</b><br>{critical[0]['message']} "
            "如果直接把该字段交给模型，会得到虚高但没有业务价值的指标。</div>",
            unsafe_allow_html=True,
        )
    st.subheader("目标泄漏风险")
    if leakage:
        display = pd.DataFrame(leakage).rename(columns={
            "column": "字段", "severity": "等级", "type": "证据类型", "accuracy": "规则匹配率", "message": "解释"
        })
        st.dataframe(display, width="stretch", hide_index=True)
    else:
        st.success("未发现高风险单字段规则。")
    st.subheader("冗余关系")
    if redundancy:
        st.dataframe(pd.DataFrame(redundancy), width="stretch", hide_index=True)
    with st.expander("查看数据质量评分扣分项"):
        st.json(quality)

with tab3:
    prompt = st.text_area("描述你的任务", f"检查数据质量，清理数据并建立 {target} 基线模型")
    if st.button("生成处理计划", type="primary"):
        try:
            planner = HybridPlanner(ollama_model) if planner_mode == "Ollama本地模型" else RulePlanner()
            st.session_state["plan"] = planner.plan(prompt, df, target, split_strategy)
            st.session_state.pop("result", None)
        except Exception as exc:
            st.error(f"规划失败：{exc}")
    plan = st.session_state.get("plan")
    if plan:
        st.subheader("结构化执行计划")
        planner_label = {"rule":"规则规划器", "ollama":"Ollama本地模型", "rule_fallback":"规则安全回退"}.get(plan.planner, plan.planner)
        st.caption(f"计划来源：{planner_label}")
        if plan.fallback_reason:
            st.info(plan.fallback_reason)
        plan_table = pd.DataFrame([
            {
                "顺序": index + 1,
                "工具": step.tool,
                "执行原因": step.reason,
                "需要确认": "是" if step.requires_confirmation else "否",
                "参数": json.dumps(step.parameters, ensure_ascii=False),
            }
            for index, step in enumerate(plan.steps)
        ])
        st.dataframe(plan_table, hide_index=True, width="stretch")
        with st.expander("查看原始JSON计划"):
            st.json(plan.to_dict())
        if plan.warnings:
            for warning in plan.warnings:
                st.warning(warning)
        st.caption("点击下方按钮即表示已检查字段排除和清洗步骤。原始CSV不会被覆盖。")
        if st.button("确认计划并执行", type="primary"):
            try:
                with st.spinner("正在执行确定性数据工具..."):
                    result = TrafficDataWorkflow(df).execute(plan)
                    output_dir = ROOT / "outputs" / "latest"
                    TrafficDataWorkflow.save_outputs(result, plan, output_dir)
                    st.session_state["result"] = result
                st.success("执行完成，结果已保存到 outputs/latest。")
            except Exception as exc:
                st.error(f"执行失败：{exc}")

with tab4:
    result = st.session_state.get("result")
    if not result:
        st.info("请先在 Agent计划 页面生成并确认执行计划。")
    else:
        before, after = result.profile_before, result.profile_after
        comparison = pd.DataFrame({
            "指标": ["数据行数", "重复记录", "缺失单元格", "字段数量"],
            "处理前": [before["rows"], before["duplicate_rows"], before["missing_cells"], before["columns"]],
            "处理后": [after["rows"], after["duplicate_rows"], after["missing_cells"], after["columns"]],
        })
        st.subheader("处理前后对比")
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.subheader("泄漏与划分方式对照实验")
        st.markdown(
            "<div class='safe-note'><b>阅读方式：</b>带泄漏字段的结果只用于证明风险。"
            "最终结论应以排除泄漏字段后的时间划分结果为准。</div>",
            unsafe_allow_html=True,
        )
        experiment_rows = []
        for item in result.experiments:
            row = {"实验": item["label"], "用途": "可信基线" if item["trusted"] else "泄漏演示"}
            if item["task"] == "classification":
                row.update({"Accuracy": item["accuracy"], "Precision": item["precision"], "Recall": item["recall"], "F1": item["f1"]})
            else:
                row.update({"R²": item["r2"], "MAE": item["mae"], "RMSE": item["rmse"]})
            experiment_rows.append(row)
        st.dataframe(pd.DataFrame(experiment_rows), hide_index=True, width="stretch")
        st.pyplot(experiment_figure(result.experiments), width="stretch")

        if result.metrics.get("validation_comparison"):
            st.subheader("验证集模型选择")
            st.caption("模型只在验证集上比较；选定后才在最终测试集评估一次。")
            selection_table = pd.DataFrame(result.metrics["validation_comparison"])
            st.dataframe(selection_table, hide_index=True, width="stretch")

        metric_cols = st.columns(4)
        if result.metrics["task"] == "classification":
            for column, key, label in zip(metric_cols, ["accuracy", "precision", "recall", "f1"], ["Accuracy", "Precision", "Recall", "F1"]):
                column.metric(label, f"{result.metrics[key]:.2%}")
            chart_left, chart_right = st.columns([1, 1.4])
            with chart_left:
                st.pyplot(confusion_figure(result.metrics), width="stretch")
            with chart_right:
                st.markdown("### 推荐结论")
                st.write(
                    f"排除泄漏字段并采用{('时间顺序' if result.metrics['split_strategy']=='time' else '随机')}划分后，"
                    f"模型在 {result.metrics['test_rows']} 条测试记录上取得 Recall {result.metrics['recall']:.2%}。"
                )
                st.write("该结果用于证明治理后的数据仍具备下游建模价值，不用于宣称真实道路部署效果。")
                if result.metrics.get("top_features"):
                    importance = pd.DataFrame(result.metrics["top_features"]).set_index("feature")
                    st.markdown("### 模型关注的主要特征")
                    st.bar_chart(importance["importance"])
        else:
            for column, key, label in zip(metric_cols[:3], ["r2", "mae", "rmse"], ["R²", "MAE", "RMSE"]):
                column.metric(label, f"{result.metrics[key]:.3f}")

        st.subheader("Agent执行轨迹")
        trace_df = pd.DataFrame(result.logs).rename(columns={"time":"时间", "action":"工具", "detail":"结果", "affected":"影响数量"})
        st.dataframe(trace_df, hide_index=True, width="stretch")
        cleaned_csv = result.cleaned_data.to_csv(index=False).encode("utf-8-sig")
        download_cols = st.columns(4)
        download_cols[0].download_button("下载清洗数据", cleaned_csv, "cleaned_traffic_data.csv", "text/csv", width="stretch")
        report_file = ROOT / "outputs" / "latest" / "analysis_report.md"
        if report_file.exists():
            download_cols[1].download_button("下载Markdown报告", report_file.read_bytes(), "analysis_report.md", "text/markdown", width="stretch")
        html_file = ROOT / "outputs" / "latest" / "analysis_report.html"
        if html_file.exists():
            download_cols[2].download_button("下载HTML报告", html_file.read_bytes(), "analysis_report.html", "text/html", width="stretch")
        dictionary_file = ROOT / "outputs" / "latest" / "data_dictionary.csv"
        if dictionary_file.exists():
            download_cols[3].download_button("下载数据字典", dictionary_file.read_bytes(), "data_dictionary.csv", "text/csv", width="stretch")
