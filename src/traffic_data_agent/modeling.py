from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _prepare_xy(df: pd.DataFrame, target: str, excluded: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    drop = set(excluded) | {target}
    feature_columns = [column for column in df.columns if column not in drop]
    x = df[feature_columns].copy()
    if "timestamp" in x.columns:
        timestamp = pd.to_datetime(x.pop("timestamp"), errors="coerce")
        x["timestamp_hour"] = timestamp.dt.hour
        x["timestamp_dayofweek"] = timestamp.dt.dayofweek
    return x, df[target]


def _preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric = x.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in x.columns if column not in numeric]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])


def _split(
    x: pd.DataFrame,
    y: pd.Series,
    strategy: str,
    timestamps: pd.Series | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if strategy == "time" and timestamps is not None:
        order = pd.to_datetime(timestamps, errors="coerce").sort_values().index
        x_ordered = x.loc[order]
        y_ordered = y.loc[order]
        cut = max(1, int(len(x_ordered) * 0.75))
        return x_ordered.iloc[:cut], x_ordered.iloc[cut:], y_ordered.iloc[:cut], y_ordered.iloc[cut:]
    stratify = y if y.nunique() == 2 else None
    return train_test_split(x, y, test_size=0.25, random_state=42, stratify=stratify)


def train_baseline(
    df: pd.DataFrame,
    target: str,
    excluded: list[str],
    split_strategy: str = "time",
    model_name: str = "default",
) -> dict[str, Any]:
    if target not in df.columns:
        raise KeyError(f"目标字段不存在: {target}")
    x, y = _prepare_xy(df, target, excluded)
    timestamps = df["timestamp"] if "timestamp" in df.columns else None
    x_train, x_test, y_train, y_test = _split(x, y, split_strategy, timestamps)
    classification = y.nunique(dropna=True) == 2
    if classification and model_name == "decision_tree":
        estimator = DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, random_state=42, class_weight="balanced")
        resolved_model = "decision_tree"
    elif classification:
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced")
        resolved_model = "logistic_regression"
    else:
        estimator = Ridge(alpha=1.0)
        resolved_model = "ridge"
    pipeline = Pipeline([("preprocessor", _preprocessor(x)), ("model", estimator)])
    pipeline.fit(x_train, y_train)
    prediction = pipeline.predict(x_test)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    fitted_model = pipeline.named_steps["model"]
    if hasattr(fitted_model, "coef_"):
        importance_values = np.abs(np.asarray(fitted_model.coef_).reshape(-1))
    elif hasattr(fitted_model, "feature_importances_"):
        importance_values = np.asarray(fitted_model.feature_importances_)
    else:
        importance_values = np.zeros(len(feature_names))
    top_indexes = np.argsort(importance_values)[::-1][:10]
    top_features = [
        {
            "feature": str(feature_names[index]).replace("numeric__", "").replace("categorical__", ""),
            "importance": round(float(importance_values[index]), 6),
        }
        for index in top_indexes
        if importance_values[index] > 0
    ]

    result: dict[str, Any] = {
        "task": "classification" if classification else "regression",
        "target": target,
        "model": resolved_model,
        "split_strategy": split_strategy,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_count_before_encoding": int(x.shape[1]),
        "excluded_features": list(excluded),
        "top_features": top_features,
    }
    if classification:
        predicted_label = np.asarray(prediction).astype(y.dtype)
        result.update({
            "accuracy": round(float(accuracy_score(y_test, predicted_label)), 4),
            "precision": round(float(precision_score(y_test, predicted_label, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predicted_label, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predicted_label, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, predicted_label).tolist(),
        })
    else:
        rmse = mean_squared_error(y_test, prediction) ** 0.5
        result.update({
            "mae": round(float(mean_absolute_error(y_test, prediction)), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2_score(y_test, prediction)), 4),
        })
    return result


def _split_train_validation_test(
    x: pd.DataFrame,
    y: pd.Series,
    strategy: str,
    timestamps: pd.Series | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    if strategy == "time" and timestamps is not None:
        order = pd.to_datetime(timestamps, errors="coerce").sort_values().index
        x_ordered, y_ordered = x.loc[order], y.loc[order]
        train_end = max(1, int(len(x_ordered) * 0.60))
        validation_end = max(train_end + 1, int(len(x_ordered) * 0.80))
        return (
            x_ordered.iloc[:train_end],
            x_ordered.iloc[train_end:validation_end],
            x_ordered.iloc[validation_end:],
            y_ordered.iloc[:train_end],
            y_ordered.iloc[train_end:validation_end],
            y_ordered.iloc[validation_end:],
        )

    stratify = y if y.nunique() == 2 else None
    x_train_validation, x_test, y_train_validation, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=stratify
    )
    stratify_train = y_train_validation if y_train_validation.nunique() == 2 else None
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_validation,
        y_train_validation,
        test_size=0.25,
        random_state=42,
        stratify=stratify_train,
    )
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def _evaluate_predictions(y_true: pd.Series, prediction: np.ndarray, classification: bool) -> dict[str, Any]:
    if classification:
        predicted_label = np.asarray(prediction).astype(y_true.dtype)
        return {
            "accuracy": round(float(accuracy_score(y_true, predicted_label)), 4),
            "precision": round(float(precision_score(y_true, predicted_label, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, predicted_label, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, predicted_label, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_true, predicted_label).tolist(),
        }
    return {
        "mae": round(float(mean_absolute_error(y_true, prediction)), 4),
        "rmse": round(float(mean_squared_error(y_true, prediction) ** 0.5), 4),
        "r2": round(float(r2_score(y_true, prediction)), 4),
    }


def _extract_top_features(pipeline: Pipeline) -> list[dict[str, Any]]:
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    fitted_model = pipeline.named_steps["model"]
    if hasattr(fitted_model, "coef_"):
        coefficients = np.asarray(fitted_model.coef_)
        importance_values = np.abs(coefficients).mean(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
    elif hasattr(fitted_model, "feature_importances_"):
        importance_values = np.asarray(fitted_model.feature_importances_)
    else:
        return []
    top_indexes = np.argsort(importance_values)[::-1][:10]
    return [
        {
            "feature": str(feature_names[index]).replace("numeric__", "").replace("categorical__", ""),
            "importance": round(float(importance_values[index]), 6),
        }
        for index in top_indexes
        if importance_values[index] > 0
    ]


def train_model_suite(
    df: pd.DataFrame,
    target: str,
    excluded: list[str],
    split_strategy: str = "time",
) -> dict[str, Any]:
    """Select on validation data, then evaluate the selected model once on test data."""
    x, y = _prepare_xy(df, target, excluded)
    timestamps = df["timestamp"] if "timestamp" in df.columns else None
    x_train, x_validation, x_test, y_train, y_validation, y_test = _split_train_validation_test(
        x, y, split_strategy, timestamps
    )
    classification = y.nunique(dropna=True) == 2
    if classification:
        candidates = {
            "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "svm_rbf": SVC(kernel="rbf", C=1.0, class_weight="balanced"),
            "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, random_state=42, class_weight="balanced"),
        }
    else:
        candidates = {
            "linear_regression": LinearRegression(),
            "ridge_alpha_1": Ridge(alpha=1.0),
            "ridge_alpha_10": Ridge(alpha=10.0),
        }

    fitted: dict[str, Pipeline] = {}
    comparisons: list[dict[str, Any]] = []
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", _preprocessor(x)), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        validation_metrics = _evaluate_predictions(y_validation, pipeline.predict(x_validation), classification)
        fitted[name] = pipeline
        comparisons.append({"model": name, **validation_metrics})

    if classification:
        selected = max(comparisons, key=lambda item: (item["f1"], item["recall"], item["accuracy"]))["model"]
    else:
        selected = min(comparisons, key=lambda item: (item["rmse"], item["mae"]))["model"]

    # 模型选择完成后，用训练+验证数据重训一次，测试集此前从未参与选择。
    final_pipeline = Pipeline([("preprocessor", _preprocessor(x)), ("model", candidates[selected])])
    x_train_final = pd.concat([x_train, x_validation])
    y_train_final = pd.concat([y_train, y_validation])
    final_pipeline.fit(x_train_final, y_train_final)
    test_metrics = _evaluate_predictions(y_test, final_pipeline.predict(x_test), classification)
    return {
        "task": "classification" if classification else "regression",
        "target": target,
        "model": selected,
        "selection_metric": "validation_f1" if classification else "validation_rmse",
        "split_strategy": split_strategy,
        "train_rows": int(len(x_train)),
        "validation_rows": int(len(x_validation)),
        "test_rows": int(len(x_test)),
        "feature_count_before_encoding": int(x.shape[1]),
        "excluded_features": list(excluded),
        "validation_comparison": comparisons,
        "top_features": _extract_top_features(final_pipeline),
        **test_metrics,
    }
