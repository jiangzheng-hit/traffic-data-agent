from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .agent import RulePlanner
from .pipeline import TrafficDataWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(description="交通数据质量审查与建模助手")
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", default="is_congested")
    parser.add_argument("--split", choices=["random", "time"], default="time")
    parser.add_argument("--output", default="outputs/latest")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    plan = RulePlanner().plan(f"清洗数据并预测 {args.target}", df, args.target, args.split)
    result = TrafficDataWorkflow(df).execute(plan)
    TrafficDataWorkflow.save_outputs(result, plan, Path(args.output))
    print(f"完成：{args.output}")
    print(result.metrics)


if __name__ == "__main__":
    main()

