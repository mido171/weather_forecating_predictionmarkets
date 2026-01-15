"""Report helpers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve


def write_report(
    path: Path,
    *,
    dataset_summary: dict,
    metrics: dict,
    model_summary: dict,
    feature_importance: dict | None,
    global_calibration: dict | None,
    baseline_calibration: dict | None,
    config: dict,
) -> None:
    lines = []
    lines.append("# Training Report")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append("```json")
    lines.append(_pretty_json(dataset_summary))
    lines.append("```")
    lines.append("")
    lines.append("## Model Summary")
    lines.append("```json")
    lines.append(_pretty_json(model_summary))
    lines.append("```")
    lines.append("")
    lines.append("## Metrics Summary")
    lines.append("```json")
    lines.append(_pretty_json(metrics))
    lines.append("```")
    lines.append("")
    lines.append("## Feature Importance")
    lines.append("```json")
    lines.append(_pretty_json(feature_importance or {}))
    lines.append("```")
    lines.append("")
    if global_calibration:
        lines.append("## Global Normal Calibration (2025)")
        lines.append("")
        lines.append(
            f"Calibration window: {global_calibration['calibration_window']['start']} "
            f"to {global_calibration['calibration_window']['end']}"
        )
        lines.append(f"Rows used: {global_calibration['n']}")
        lines.append("")
        lines.append("```json")
        lines.append(_pretty_json(global_calibration))
        lines.append("```")
        lines.append("")
    if baseline_calibration:
        lines.append("## Baseline Median Calibration (2021-2025)")
        lines.append("")
        lines.append(
            f"Calibration window: {baseline_calibration['calibration_window']['start']} "
            f"to {baseline_calibration['calibration_window']['end']}"
        )
        lines.append(f"Rows used: {baseline_calibration['n']}")
        lines.append("")
        lines.append("```json")
        lines.append(_pretty_json(baseline_calibration))
        lines.append("```")
        lines.append("")
    lines.append("## Config Snapshot")
    lines.append("```json")
    lines.append(_pretty_json(config))
    lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_residual_hist(path: Path, residuals: np.ndarray) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, edgecolor="black")
    plt.title("Residual Histogram")
    plt.xlabel("Residual (F)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_residual_vs_pred(path: Path, y_pred: np.ndarray, residuals: np.ndarray) -> None:
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.title("Residual vs Prediction")
    plt.xlabel("Predicted Tmax (F)")
    plt.ylabel("Residual (F)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_calibration_curve(
    path: Path,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    title: str,
) -> None:
    y_prob = np.clip(y_prob, 0.0, 1.0)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    plt.figure(figsize=(6, 4))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _pretty_json(payload: dict) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        default=_json_default,
    )


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)
