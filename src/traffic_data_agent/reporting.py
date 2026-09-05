from __future__ import annotations

from typing import Any


def build_markdown_report(
    profile_before: dict[str, Any],
    profile_after: dict[str, Any],
    leakage: list[dict[str, Any]],
    redundancy: list[dict[str, Any]],
    validation: dict[str, Any],
    metrics: dict[str, Any],
    experiments: list[dict[str, Any]],
    quality: dict[str, Any],
    logs: list[dict[str, Any]],
) -> str:
    lines = [
        "# 交通数据质量与建模报告",
        "",
        "## 数据概况",
        "",
        f"- 原始数据质量评分：{quality['score']}/100（{quality['level']}）。",
        f"- 处理前：{profile_before['rows']} 行，{profile_before['columns']} 列，"
        f"{profile_before['duplicate_rows']} 条重复记录，{profile_before['missing_cells']} 个缺失单元格。",
        f"- 处理后：{profile_after['rows']} 行，{profile_after['columns']} 列，"
        f"{profile_after['duplicate_rows']} 条重复记录，{profile_after['missing_cells']} 个缺失单元格。",
        "",
        "## 目标泄漏风险",
        "",
    ]
    lines.extend([f"- [{item['severity']}] {item['message']}" for item in leakage] or ["- 未发现高风险单字段规则。"])
    lines.extend(["", "## 冗余字段", ""])
    lines.extend([f"- {item['message']}" for item in redundancy] or ["- 未发现明确冗余关系。"])
    lines.extend(["", "## 数据验证", ""])
    lines.extend([f"- {'通过' if item['passed'] else '未通过'}：{item['name']}，{item['detail']}" for item in validation["checks"]])
    lines.extend(["", "## 对照实验", ""])
    for item in experiments:
        if item["task"] == "classification":
            summary = f"Accuracy {item['accuracy']:.2%}，Recall {item['recall']:.2%}，F1 {item['f1']:.2%}"
        else:
            summary = f"R² {item['r2']:.3f}，MAE {item['mae']:.3f}，RMSE {item['rmse']:.3f}"
        lines.append(f"- {item['label']}：{summary}。{item['interpretation']}")
    lines.extend(["", "## 推荐模型结果", "", f"```json\n{_pretty(metrics)}\n```", "", "## Agent执行轨迹", ""])
    lines.extend([f"- {item['time']} {item['action']}：{item['detail']}（影响 {item['affected']}）" for item in logs])
    lines.extend([
        "",
        "## 使用限制",
        "",
        "样例数据规模较小，指标用于课程学习与工作流验证。上线或真实道路应用前，应使用跨时段、跨道路数据进行外部验证。",
    ])
    return "\n".join(lines)


def build_html_report(
    markdown_summary: str,
    quality: dict[str, Any],
    experiments: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> str:
    import html

    experiment_rows = "".join(
        f"<tr><td>{html.escape(item['label'])}</td>"
        f"<td>{item.get('accuracy', item.get('r2', ''))}</td>"
        f"<td>{item.get('recall', item.get('mae', ''))}</td>"
        f"<td>{'推荐基线' if item['trusted'] and item['id'].endswith('time') else ('参考' if item['trusted'] else '泄漏演示')}</td></tr>"
        for item in experiments
    )
    validation_rows = "".join(
        f"<tr><td>{html.escape(item['model'])}</td>"
        f"<td>{item.get('accuracy', item.get('r2', ''))}</td>"
        f"<td>{item.get('recall', item.get('mae', ''))}</td>"
        f"<td>{item.get('f1', item.get('rmse', ''))}</td></tr>"
        for item in metrics.get("validation_comparison", [])
    )
    content_html = _markdown_to_html(markdown_summary)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>交通数据质量与建模报告</title>
<style>
body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:0;background:#f5f8fa;color:#23313a}}
.page{{max-width:980px;margin:28px auto;background:white;padding:38px 48px;box-shadow:0 8px 30px #17324d18}}
h1{{color:#17324d}} .score{{font-size:46px;font-weight:700;color:#0d7c86}}
table{{border-collapse:collapse;width:100%;margin:18px 0}} th,td{{border:1px solid #d9e2e7;padding:10px;text-align:left}}
th{{background:#17324d;color:white}} img{{max-width:100%;margin:12px 0}}
pre{{white-space:pre-wrap;line-height:1.55;background:#f7fafb;padding:18px;border-left:4px solid #0d7c86;overflow:auto}}
li{{margin:7px 0;line-height:1.55}} h2{{color:#17324d;margin-top:30px}} code{{font-family:Consolas,monospace}}
</style></head><body><main class="page"><h1>交通数据质量与建模报告</h1>
<div class="score">{quality['score']}<small>/100</small></div><p>{html.escape(quality['level'])}</p>
<h2>对照实验</h2><table><thead><tr><th>实验</th><th>Accuracy 或 R²</th><th>Recall 或 MAE</th><th>用途</th></tr></thead>
<tbody>{experiment_rows}</tbody></table>
<h2>验证集模型选择</h2><p>以下结果只用于选择模型，最终测试集此前未参与比较。</p>
<table><thead><tr><th>模型</th><th>Accuracy 或 R²</th><th>Recall 或 MAE</th><th>F1 或 RMSE</th></tr></thead>
<tbody>{validation_rows}</tbody></table>
<img src="figures/experiments.png" alt="对照实验图"><img src="figures/traffic_relationships.png" alt="交通关系图">
<h2>完整结果</h2>{content_html}</main></body></html>"""


def _markdown_to_html(text: str) -> str:
    import html

    output: list[str] = []
    in_list = False
    in_code = False
    code_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            if in_code:
                output.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines = []
                in_code = False
            else:
                if in_list:
                    output.append("</ul>")
                    in_list = False
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            if in_list:
                output.append("</ul>")
                in_list = False
            output.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            if not in_list:
                output.append("<ul>")
                in_list = True
            output.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.strip():
            if in_list:
                output.append("</ul>")
                in_list = False
            output.append(f"<p>{html.escape(line)}</p>")
        elif in_list:
            output.append("</ul>")
            in_list = False
    if in_list:
        output.append("</ul>")
    if in_code:
        output.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
    return "".join(output)


def _pretty(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)
