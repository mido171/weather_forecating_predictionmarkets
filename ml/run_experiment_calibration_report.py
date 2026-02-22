"""Generate calibration report for an experiment predictions file."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class EventSpec:
    name: str
    kind: str
    lt: float | None = None
    ge: float | None = None
    start: float | None = None
    end: float | None = None


DEFAULT_EVENTS = [
    EventSpec("lt_52", "threshold", lt=52),
    EventSpec("lt_70", "threshold", lt=70),
    EventSpec("lt_75", "threshold", lt=75),
    EventSpec("ge_85", "threshold", ge=85),
    EventSpec("ge_90", "threshold", ge=90),
    EventSpec("range_80_84", "range", start=80, end=84),
    EventSpec("range_85_89", "range", start=85, end=89),
]


def _read_config_csv_path(experiment_dir: Path) -> Path | None:
    cfg_path = experiment_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        return None
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    csv_path = data.get("data", {}).get("csv_path")
    if not csv_path:
        return None
    return Path(csv_path)


def _load_truth(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "target_tmax_f" not in df.columns and "actual_tmax_f" not in df.columns:
        raise ValueError("Truth column not found in dataset CSV.")
    truth_col = "target_tmax_f" if "target_tmax_f" in df.columns else "actual_tmax_f"
    df = df[["station_id", "target_date_local", "asof_utc", truth_col]].copy()
    df = df.rename(columns={truth_col: "y_true"})
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
    return df


def _load_predictions(pred_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(pred_path)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
    return df


def _parse_pmf(json_series: pd.Series) -> np.ndarray:
    pmf_list = json_series.apply(lambda s: json.loads(s) if isinstance(s, str) else s)
    return np.vstack(pmf_list.to_list()).astype(float)


def _pmf_support(df_pred: pd.DataFrame) -> tuple[int, int]:
    support_min = int(df_pred["support_min_f"].dropna().iloc[0])
    support_max = int(df_pred["support_max_f"].dropna().iloc[0])
    return support_min, support_max


def _pmf_log_loss(y_true: np.ndarray, pmf: np.ndarray, support_min: int) -> float:
    idx = (y_true.astype(int) - support_min).clip(0, pmf.shape[1] - 1)
    probs = pmf[np.arange(len(y_true)), idx]
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.mean(np.log(probs)))


def _pmf_entropy(pmf: np.ndarray) -> float:
    p = np.clip(pmf, 1e-12, 1.0)
    return float(np.mean(-np.sum(p * np.log(p), axis=1)))


def _pmf_variance(pmf: np.ndarray, support_min: int) -> float:
    values = np.arange(pmf.shape[1]) + support_min
    mean = np.sum(pmf * values[None, :], axis=1)
    var = np.sum(pmf * (values[None, :] - mean[:, None]) ** 2, axis=1)
    return float(np.mean(var))


def _cdf_from_pmf(pmf: np.ndarray) -> np.ndarray:
    return np.cumsum(pmf, axis=1)


def _pit_values(y_true: np.ndarray, pmf: np.ndarray, support_min: int) -> np.ndarray:
    idx = (y_true.astype(int) - support_min).clip(0, pmf.shape[1] - 1)
    cdf = _cdf_from_pmf(pmf)
    pit = cdf[np.arange(len(y_true)), idx] - 0.5 * pmf[np.arange(len(y_true)), idx]
    return pit


def _pit_stats(pit: np.ndarray) -> dict[str, Any]:
    pit = pit[np.isfinite(pit)]
    if pit.size == 0:
        return {}
    hist, edges = np.histogram(pit, bins=10, range=(0.0, 1.0))
    expected = len(pit) / 10.0
    return {
        "count": int(len(pit)),
        "mean": float(np.mean(pit)),
        "std": float(np.std(pit)),
        "hist_bins": hist.tolist(),
        "hist_edges": edges.tolist(),
        "chi2": float(np.sum((hist - expected) ** 2 / max(expected, 1.0))),
    }


def _event_probs_from_pmf(
    pmf: np.ndarray,
    support_min: int,
    support_max: int,
    spec: EventSpec,
) -> np.ndarray:
    values = np.arange(pmf.shape[1]) + support_min
    if spec.kind == "threshold":
        if spec.lt is not None:
            mask = values < spec.lt
        elif spec.ge is not None:
            mask = values >= spec.ge
        else:
            raise ValueError("Invalid threshold spec.")
    elif spec.kind == "range":
        mask = (values >= spec.start) & (values <= spec.end)
    else:
        raise ValueError(f"Unknown spec kind: {spec.kind}")
    return pmf[:, mask].sum(axis=1)


def _event_indicator(y_true: np.ndarray, spec: EventSpec) -> np.ndarray:
    if spec.kind == "threshold":
        if spec.lt is not None:
            return (y_true < spec.lt).astype(int)
        if spec.ge is not None:
            return (y_true >= spec.ge).astype(int)
    if spec.kind == "range":
        return ((y_true >= spec.start) & (y_true <= spec.end)).astype(int)
    raise ValueError(f"Unknown spec kind: {spec.kind}")


def _brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    probs = np.clip(probs, 0.0, 1.0)
    return float(np.mean((y_true - probs) ** 2))


def _log_loss_binary(y_true: np.ndarray, probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _reliability_table(probs: np.ndarray, y_true: np.ndarray, bins: int = 10) -> dict[str, Any]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(probs, edges, right=True) - 1
    idx = np.clip(idx, 0, bins - 1)
    rows = []
    ece = 0.0
    mce = 0.0
    total = len(probs)
    for b in range(bins):
        mask = idx == b
        if not np.any(mask):
            rows.append({"bin": b, "count": 0, "avg_pred": None, "emp_rate": None})
            continue
        avg_pred = float(np.mean(probs[mask]))
        emp_rate = float(np.mean(y_true[mask]))
        rows.append(
            {"bin": b, "count": int(mask.sum()), "avg_pred": avg_pred, "emp_rate": emp_rate}
        )
        gap = abs(avg_pred - emp_rate)
        ece += (mask.sum() / total) * gap
        mce = max(mce, gap)
    return {"bins": rows, "ece": float(ece), "mce": float(mce)}


def _interval_coverage(
    pmf: np.ndarray, support_min: int, y_true: np.ndarray, levels: list[float]
) -> dict[str, Any]:
    values = np.arange(pmf.shape[1]) + support_min
    cdf = _cdf_from_pmf(pmf)
    results = {}
    for level in levels:
        lower_q = (1.0 - level) / 2.0
        upper_q = 1.0 - lower_q
        lower_vals = values[np.argmax(cdf >= lower_q, axis=1)]
        upper_vals = values[np.argmax(cdf >= upper_q, axis=1)]
        within = (y_true >= lower_vals) & (y_true <= upper_vals)
        width = upper_vals - lower_vals
        results[f"p{int(level*100)}"] = {
            "coverage": float(np.mean(within)),
            "avg_width": float(np.mean(width)),
        }
    return results


def build_report(
    pred_path: Path,
    dataset_csv: Path,
    out_dir: Path,
    events: list[EventSpec],
) -> dict[str, Any]:
    df_pred = _load_predictions(pred_path)
    df_truth = _load_truth(dataset_csv)
    merged = df_pred.merge(df_truth, on=["station_id", "target_date_local", "asof_utc"], how="left")
    merged = merged.dropna(subset=["y_true", "p_temp_json"])
    if merged.empty:
        raise ValueError("No rows after merging predictions with truth.")

    pmf = _parse_pmf(merged["p_temp_json"])
    y_true = merged["y_true"].to_numpy(dtype=float)
    support_min, support_max = _pmf_support(merged)

    report: dict[str, Any] = {
        "rows": int(len(merged)),
        "support_min_f": support_min,
        "support_max_f": support_max,
        "log_loss_pmf": _pmf_log_loss(y_true, pmf, support_min),
        "entropy_mean": _pmf_entropy(pmf),
        "variance_mean": _pmf_variance(pmf, support_min),
        "pit": _pit_stats(_pit_values(y_true, pmf, support_min)),
        "intervals": _interval_coverage(pmf, support_min, y_true, levels=[0.5, 0.8, 0.9]),
        "events": {},
    }

    for spec in events:
        probs = _event_probs_from_pmf(pmf, support_min, support_max, spec)
        y_event = _event_indicator(y_true, spec)
        report["events"][spec.name] = {
            "brier": _brier(y_event, probs),
            "log_loss": _log_loss_binary(y_event, probs),
            "reliability": _reliability_table(probs, y_event, bins=10),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(out_dir / "calibration_report.md", report)
    _write_reliability_csv(out_dir / "calibration_reliability.csv", report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Calibration Report", ""]
    lines.append(f"- rows={report['rows']}")
    lines.append(f"- log_loss_pmf={report['log_loss_pmf']:.4f}")
    lines.append(f"- entropy_mean={report['entropy_mean']:.4f}")
    lines.append(f"- variance_mean={report['variance_mean']:.4f}")
    pit = report.get("pit", {})
    if pit:
        lines.append(f"- PIT mean={pit.get('mean'):.4f} std={pit.get('std'):.4f} chi2={pit.get('chi2'):.2f}")
    lines.append("")
    lines.append("## Interval Coverage")
    for key, payload in report.get("intervals", {}).items():
        lines.append(f"- {key}: coverage={payload['coverage']:.3f} avg_width={payload['avg_width']:.3f}")
    lines.append("")
    lines.append("## Events")
    for name, payload in report.get("events", {}).items():
        lines.append(f"- {name}: brier={payload['brier']:.4f} log_loss={payload['log_loss']:.4f} ece={payload['reliability']['ece']:.4f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_reliability_csv(path: Path, report: dict[str, Any]) -> None:
    rows = []
    for name, payload in report.get("events", {}).items():
        rel = payload.get("reliability", {})
        for bin_row in rel.get("bins", []):
            rows.append(
                {
                    "event": name,
                    "bin": bin_row.get("bin"),
                    "count": bin_row.get("count"),
                    "avg_pred": bin_row.get("avg_pred"),
                    "emp_rate": bin_row.get("emp_rate"),
                    "ece": rel.get("ece"),
                    "mce": rel.get("mce"),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate calibration report for an experiment.")
    parser.add_argument("--experiment-dir", required=True, help="Path to experiment directory.")
    parser.add_argument("--dataset-csv", help="Override dataset CSV path.")
    parser.add_argument("--out-dir", help="Override output directory (default: experiment dir).")
    args = parser.parse_args(argv)

    experiment_dir = Path(args.experiment_dir)
    pred_path = experiment_dir / "predictions_test.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions_test.parquet not found in {experiment_dir}")

    dataset_csv = Path(args.dataset_csv) if args.dataset_csv else _read_config_csv_path(experiment_dir)
    if dataset_csv is None or not dataset_csv.exists():
        raise FileNotFoundError("Dataset CSV not found; pass --dataset-csv or check config_resolved.yaml.")

    out_dir = Path(args.out_dir) if args.out_dir else experiment_dir
    build_report(pred_path, dataset_csv, out_dir, DEFAULT_EVENTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
