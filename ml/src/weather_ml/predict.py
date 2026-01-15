"""Prediction CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np

from weather_ml import calibration
from weather_ml import config as config_module
from weather_ml import dataset
from weather_ml import distribution
from weather_ml import features
from weather_ml import models_sigma

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Weather ML prediction helper.")
    parser.add_argument("--run-dir", required=True, help="Path to trained run dir.")
    parser.add_argument("--csv", required=True, help="Input CSV for inference.")
    parser.add_argument(
        "--output",
        help="Output path for predictions (parquet or csv).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_resolved.yaml in {run_dir}")

    config = config_module.load_config(config_path)
    config = config_module.resolve_paths(config, repo_root=_resolve_repo_root())

    mean_model = joblib.load(run_dir / "mean_model.joblib")
    sigma_model = joblib.load(run_dir / "sigma_model.joblib")
    feature_state = joblib.load(run_dir / "feature_state.joblib")
    calibrators = {}
    calibrator_path = run_dir / "calibrators.joblib"
    if calibrator_path.exists():
        calibrators = joblib.load(calibrator_path)

    df = dataset.load_csv(args.csv)
    _validate_input_schema(df, config)

    X_df, _, _ = features.build_features(df, config=config, fit_state=feature_state)
    X = X_df.to_numpy(dtype=float)

    mu = mean_model.predict(X)
    sigma = models_sigma.predict_sigma(
        sigma_model,
        X=X,
        eps=config.models.sigma.eps,
        sigma_floor=config.models.sigma.sigma_floor,
    )

    pmf = np.vstack(
        [
            distribution.normal_integer_pmf(
                float(mu_val),
                float(sigma_val),
                support_min=config.distribution.support_min_f,
                support_max=config.distribution.support_max_f,
            )
            for mu_val, sigma_val in zip(mu, sigma)
        ]
    )

    bin_probs = _bin_probabilities(
        pmf,
        config.distribution.support_min_f,
        config.calibration.bins_to_calibrate,
    )
    if calibrators:
        bin_probs = calibration.apply_calibrators(bin_probs, calibrators)

    output_path = Path(args.output) if args.output else run_dir / "predictions.parquet"
    _write_predictions(
        output_path,
        df,
        mu,
        sigma,
        pmf,
        bin_probs,
        config.distribution.support_min_f,
    )
    LOGGER.info("Predictions written to %s", output_path)
    return 0


def _validate_input_schema(df, config) -> None:
    required = set(config.features.base_features)
    required.update({"station_id", "target_date_local", "asof_utc"})
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for prediction: {missing}")


def _bin_probabilities(
    pmf: np.ndarray, support_min: int, bin_specs: list[dict]
) -> dict[str, np.ndarray]:
    probs: dict[str, np.ndarray] = {}
    for spec in bin_specs:
        name = spec.get("name")
        if not name:
            continue
        if spec.get("type") == "threshold":
            if "lt" in spec:
                cutoff = int(spec["lt"])
                idx = cutoff - support_min
                probs[name] = pmf[:, :idx].sum(axis=1)
            elif "ge" in spec:
                cutoff = int(spec["ge"])
                idx = cutoff - support_min
                probs[name] = pmf[:, idx:].sum(axis=1)
        elif spec.get("type") == "range":
            start = int(spec["start"])
            end = int(spec["end"])
            start_idx = max(start - support_min, 0)
            end_idx = min(end - support_min + 1, pmf.shape[1])
            probs[name] = pmf[:, start_idx:end_idx].sum(axis=1)
    return probs


def _write_predictions(
    path: Path,
    df,
    mu: np.ndarray,
    sigma: np.ndarray,
    pmf: np.ndarray,
    bin_probs: dict[str, np.ndarray],
    support_min: int,
) -> None:
    records = df[["station_id", "target_date_local", "asof_utc"]].copy()
    records["mu_hat_f"] = mu
    records["sigma_hat_f"] = sigma
    records["p_temp_json"] = [
        json.dumps(row.tolist(), separators=(",", ":"), ensure_ascii=True) for row in pmf
    ]
    records["p_bins_json"] = [
        json.dumps(
            {name: float(prob[idx]) for name, prob in bin_probs.items()},
            separators=(",", ":"),
            ensure_ascii=True,
        )
        for idx in range(len(records))
    ]
    records["support_min_f"] = support_min
    records["support_max_f"] = support_min + pmf.shape[1] - 1
    if path.suffix.lower() == ".csv":
        records.to_csv(path, index=False)
    else:
        records.to_parquet(path, index=False, engine="pyarrow")


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    raise SystemExit(main())
