# AI Native 交通数据质量审查与建模助手

这是在《交通数据处理方法》课程作业基础上改造的本地项目。系统读取交通运行 CSV，完成数据画像、重复值和缺失值检查、目标泄漏诊断、冗余字段分析、可追溯清洗、基线建模与报告导出。

项目的重点不是让大模型直接修改数据，而是让 Agent 把自然语言需求转换为受约束的执行计划，再调用确定性的 Python 工具。高风险处理需要用户确认，模型指标和报告数值均来自实际计算结果。

## 核心亮点

- 自动发现重复记录、缺失值、字段类型和目标分布。
- 搜索单字段阈值规则，识别潜在目标泄漏。
- 识别时间派生字段和类别字段间的固定映射。
- 支持规则规划器，默认完全离线运行。
- 可选连接本机 Ollama，把自然语言转换为结构化工具计划。
- 比较随机分层划分与时间顺序划分的模型表现。
- 自动生成数据质量评分和字段风险证据。
- 导出清洗数据、执行计划、处理日志、指标、图表、Markdown和HTML报告。

## 已在样例数据中发现的问题

- 原始数据 726 行、22 个字段，存在 6 条完全重复记录。
- `weather`、`occupancy`、`signal_delay_s` 合计存在 30 处缺失值。
- `is_congested` 与规则 `speed_kmh < 35` 在数据中完全一致。使用 `speed_kmh` 预测拥堵会造成严重目标泄漏。
- `congestion_level` 与速度区间和拥堵状态高度相关，也不应作为拥堵分类输入。
- `hour` 可由 `timestamp` 直接生成；`road_id` 与 `road_name`、`district`、`speed_limit` 存在固定映射。

## 项目结构

```text
数据集处理/
├─ 启动项目.bat          # 推荐：双击启动网页界面
├─ 生成演示结果.bat      # 推荐：双击生成命令行演示结果
├─ 启动项目.ps1          # PowerShell 启动脚本，由同名 bat 调用
├─ 生成演示结果.ps1      # PowerShell 演示脚本，由同名 bat 调用
├─ app.py                # Streamlit 网页入口
├─ requirements.txt      # Python 依赖清单
├─ pyproject.toml        # 项目及测试配置
├─ .gitignore            # Git 忽略规则
├─ data/raw/             # 原始 CSV，不会被覆盖
├─ docs/                 # 原始作业与项目讲解材料
├─ outputs/              # 清洗数据、指标、图表和报告
├─ src/traffic_data_agent/ # Agent、数据审查和建模源码
└─ tests/                # 自动化测试
```

运行 Python 或测试后，系统可能再次生成 `__pycache__`、`.pytest_cache` 和
`*.egg-info`。它们只是缓存或安装元数据，不属于项目核心内容，已经加入
`.gitignore`，删除也不会影响项目。

## 本地运行

建议使用 Python 3.10 或更高版本。

最简单的方式是直接双击 `启动项目.bat`。它会调用 PowerShell 脚本，并临时绕过
Windows 对本地脚本的执行策略；不会修改系统的永久执行策略。

`.ps1` 是 PowerShell 脚本扩展名。Windows 默认通常不会像 `.exe` 一样双击执行，
也可能受到执行策略限制。如果希望手动运行，可在 PowerShell 中执行：

```powershell
& ".\启动项目.ps1"
```

若提示“禁止运行脚本”，可只对本次调用临时放行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\启动项目.ps1"
```

也可以创建独立虚拟环境：

```powershell
cd traffic-data-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
streamlit run app.py
```

从 GitHub 获取项目时：

```powershell
git clone https://github.com/你的用户名/traffic-data-agent.git
cd traffic-data-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
streamlit run app.py
```

如果当前环境已经安装依赖，可以直接运行：

```powershell
streamlit run app.py
```

## 可选 Ollama 模式

规则规划器不依赖大模型。需要体验本地大模型规划时：

1. 安装并启动 Ollama。
2. 下载一个指令模型，例如 `qwen2.5:3b`。
3. 在页面侧栏选择 Ollama，并填写模型名称。

Ollama 只生成结构化计划，真正的数据处理仍由白名单 Python 工具执行。若模型输出不合规，系统会拒绝计划并回退到规则规划器。

## 命令行演示

```powershell
python -m traffic_data_agent.cli --input data/raw/traffic_ml_homework_dataset.csv --target is_congested
```

运行结果保存在 `outputs/latest/`。

## 演示重点

使用同一决策树进行对照：包含泄漏字段时四项分类指标均为100%；排除泄漏与冗余字段后，随机划分Accuracy为83.89%，时间划分为87.78%。正式流程按训练集、验证集、测试集进行时间顺序划分，在验证集比较Logistic、SVM和决策树后选择逻辑回归，最终测试集Accuracy为93.75%、Recall为98.55%。

项目讲解材料：

- `docs/architecture.md`：系统分层与安全边界。
- `docs/demo_script.md`：三分钟演示流程。
- `docs/interview_guide.md`：面试讲解与追问。
- `docs/resume_copy.md`：简历长短两版文案。

## 测试

```powershell
pytest -q
```

## 安全与边界

- 不覆盖原始 CSV。
- 不执行大模型生成的任意代码。
- Agent 只能调用白名单工具。
- 数据删除、字段排除和建模都在用户确认后执行。
- 样例数据量较小，结果用于课程学习和项目演示，不代表真实道路部署效果。
