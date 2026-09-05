from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"navy": "#17324D", "teal": "#0D7C86", "light": "#D9EAEC", "red": "#C94C4C"}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlecolor": COLORS["navy"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def missing_figure(df: pd.DataFrame):
    _style()
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14)
        ax.axis("off")
    else:
        ax.bar(missing.index, missing.values, color=COLORS["teal"])
        ax.set_title("Missing values by field")
        ax.set_ylabel("Missing cells")
        ax.tick_params(axis="x", rotation=25)
        for index, value in enumerate(missing.values):
            ax.text(index, value + 0.2, str(int(value)), ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def traffic_relationship_figure(df: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    if {"volume", "speed_kmh", "is_congested"}.issubset(df.columns):
        colors = np.where(df["is_congested"].astype(int) == 1, COLORS["red"], COLORS["teal"])
        axes[0].scatter(df["volume"], df["speed_kmh"], c=colors, alpha=0.55, s=18, edgecolors="none")
        axes[0].set_title("Volume and speed")
        axes[0].set_xlabel("Traffic volume")
        axes[0].set_ylabel("Speed (km/h)")
        axes[0].axhline(35, color=COLORS["red"], linestyle="--", linewidth=1, label="35 km/h threshold")
        axes[0].legend(frameon=False, fontsize=8)
    if {"weather", "speed_kmh"}.issubset(df.columns):
        order = df.groupby("weather")["speed_kmh"].median().sort_values().index
        data = [df.loc[df["weather"] == item, "speed_kmh"].dropna() for item in order]
        axes[1].boxplot(data, tick_labels=order, patch_artist=True, boxprops={"facecolor": COLORS["light"]})
        axes[1].set_title("Speed by weather")
        axes[1].set_xlabel("Weather")
        axes[1].set_ylabel("Speed (km/h)")
    fig.tight_layout()
    return fig


def experiment_figure(experiments: list[dict[str, Any]]):
    _style()
    classification = experiments and experiments[0]["task"] == "classification"
    metrics = ["accuracy", "recall", "f1"] if classification else ["r2"]
    labels = [item["label"] for item in experiments]
    x = np.arange(len(labels))
    width = 0.22 if len(metrics) > 1 else 0.5
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    for index, metric in enumerate(metrics):
        offset = (index - (len(metrics) - 1) / 2) * width
        values = [item[metric] for item in experiments]
        bars = ax.bar(x + offset, values, width, label=metric.upper())
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{value:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["Leaky random", "Safe random", "Safe time"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Metric")
    ax.set_title("Why the highest score may be misleading")
    ax.legend(frameon=False, ncol=len(metrics), loc="lower left")
    fig.tight_layout()
    return fig


def confusion_figure(metrics: dict[str, Any]):
    _style()
    matrix = np.asarray(metrics.get("confusion_matrix", []))
    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    if matrix.shape == (2, 2):
        image = ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
        ax.set_title("Safe time-split confusion matrix")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.axis("off")
    fig.tight_layout()
    return fig


def save_figures(df: pd.DataFrame, experiments: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "missing": missing_figure(df),
        "traffic_relationships": traffic_relationship_figure(df),
        "experiments": experiment_figure(experiments),
        "confusion_matrix": confusion_figure(experiments[-1]),
    }
    paths: dict[str, str] = {}
    for name, figure in figures.items():
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths[name] = str(path)
    return paths
