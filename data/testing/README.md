# 公开测试数据集

本目录保留用于演示和回归测试的公开 CSV，不包含用户课程作业、个人数据或机密数据。

| 文件 | 推荐目标字段 | 推荐数据划分 | 用途 |
| --- | --- | --- | --- |
| `bike_sharing_hourly.csv` | `cnt` | 时间顺序 | 共享单车小时需求预测；含日期、时段、天气与季节特征。 |
| `air_quality.csv` | `CO(GT)` 或 `NO2(GT)` | 时间顺序 | 传感器时间序列；用于演示缺失值识别和清洗。 |
| `adult_income.csv` | `income` | 随机划分 | 非交通分类表格；用于验证工具不依赖固定课程字段。 |

## 来源与许可

- 共享单车：UCI [Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset)，CC BY 4.0。
- 空气质量：UCI [Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air)，CC BY 4.0。
- 成人收入：UCI [Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)。

`air_quality.csv` 已由原始分号分隔格式转换为标准逗号分隔 CSV，便于直接上传。数据只用于学习、演示和自动化测试；使用前请阅读原始数据集说明和许可证。
