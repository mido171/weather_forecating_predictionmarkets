import json
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

import lightgbm as lgb


def _compute_metrics(y_true: np.ndarray, p_pred: np.ndarray, threshold: float) -> Dict[str, float]:
    y_hat = (p_pred >= threshold).astype(int)
    acc = accuracy_score(y_true, y_hat)
    bal = balanced_accuracy_score(y_true, y_hat)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", pos_label=1)
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    try:
        auc = roc_auc_score(y_true, p_pred)
    except ValueError:
        auc = float("nan")
    try:
        brier = brier_score_loss(y_true, p_pred)
    except ValueError:
        brier = float("nan")
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal),
        "yes_precision": float(prec),
        "yes_recall": float(rec),
        "yes_f1": float(f1),
        "roc_auc": float(auc),
        "brier": float(brier),
        "cm_tn": int(cm[0, 0]),
        "cm_fp": int(cm[0, 1]),
        "cm_fn": int(cm[1, 0]),
        "cm_tp": int(cm[1, 1]),
    }


def _select_threshold(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    best_t = 0.5
    best_acc = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        acc = accuracy_score(y_true, p_pred >= t)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class TCN(nn.Module):
    def __init__(self, n_features: int, n_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, n_channels, kernel_size=3, padding=2, dilation=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(n_channels, n_channels, kernel_size=3, padding=4, dilation=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(n_channels, n_channels, kernel_size=3, padding=8, dilation=4),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.net(x)
        x = self.head(x)
        return x.squeeze(-1)


class SimpleTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x.squeeze(-1)


def _train_torch_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, pos_weight: float) -> nn.Module:
    device = torch.device("cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_state = None
    best_acc = -1.0
    patience = 5
    patience_left = patience

    for _ in range(1, 31):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                prob = torch.sigmoid(logits).cpu().numpy()
                preds.append(prob)
                trues.append(yb.numpy())
        p_val = np.concatenate(preds)
        y_val = np.concatenate(trues)
        acc = accuracy_score(y_val, p_val >= 0.5)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_torch(model: nn.Module, data: np.ndarray) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(SequenceDataset(data, np.zeros(data.shape[0])), batch_size=128, shuffle=False)
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb)
            preds.append(torch.sigmoid(logits).numpy())
    return np.concatenate(preds)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    data_base = repo / "data" / "tmax_hit1830"
    out_base = repo / "artifacts" / "experiments" / "early_maxout_strategy"
    reports_dir = repo / "reports"

    df = pd.read_parquet(data_base / "features.parquet")
    df["year"] = pd.to_datetime(df["target_date_local"]).dt.year
    df["split"] = np.where(df["year"] <= 2019, "train", np.where(df["year"] <= 2022, "val", np.where(df["year"] <= 2025, "test", "future")))

    feature_cols = [c for c in df.columns if c not in {"target_date_local", "cutoff_utc", "y_hit_by_cutoff", "year", "split", "tmax_full"}]
    X = df[feature_cols]
    y = df["y_hit_by_cutoff"].to_numpy()

    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"

    imputer_main = SimpleImputer(strategy="median")
    X_train = imputer_main.fit_transform(X[train_mask])
    X_val = imputer_main.transform(X[val_mask])
    X_test = imputer_main.transform(X[test_mask])

    y_train = y[train_mask.to_numpy()]
    y_val = y[val_mask.to_numpy()]
    y_test = y[test_mask.to_numpy()]

    # Load sequences from window12h_5min
    window_dir = data_base / "window12h_5min"
    seqs = []
    for day in df["target_date_local"]:
        arr = np.load(window_dir / f"{day}.npy")
        seq_vals = pd.Series(arr).interpolate(limit_direction="both").to_numpy()
        if not np.isfinite(seq_vals).any():
            seq_vals[:] = 0.0
        seq_delta = np.diff(seq_vals, prepend=seq_vals[0])
        seq_roll = pd.Series(seq_vals).rolling(3, min_periods=1).mean().to_numpy()
        seqs.append(np.stack([seq_vals, seq_delta, seq_roll], axis=1))
    seqs = np.stack(seqs)

    seq_train = seqs[train_mask.to_numpy()]
    seq_val = seqs[val_mask.to_numpy()]
    seq_test = seqs[test_mask.to_numpy()]

    seq_mean = seq_train.mean(axis=(0, 1), keepdims=True)
    seq_std = seq_train.std(axis=(0, 1), keepdims=True) + 1e-6
    seq_train = (seq_train - seq_mean) / seq_std
    seq_val = (seq_val - seq_mean) / seq_std
    seq_test = (seq_test - seq_mean) / seq_std

    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    train_loader = DataLoader(SequenceDataset(seq_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(SequenceDataset(seq_val, y_val), batch_size=128, shuffle=False)

    # EXP6 Transformer
    transformer = SimpleTransformer(n_features=seq_train.shape[2])
    transformer = _train_torch_model(transformer, train_loader, val_loader, pos_weight)
    p_train_tr = _predict_torch(transformer, seq_train)
    p_val_tr = _predict_torch(transformer, seq_val)
    p_test_tr = _predict_torch(transformer, seq_test)
    t_tr = _select_threshold(y_val, p_val_tr)

    preds_tr = df[["target_date_local", "cutoff_utc", "y_hit_by_cutoff", "temp_max_sofar", "slope_60m", "drop_from_max", "split"]].copy()
    preds_tr.loc[train_mask, "p_pred"] = p_train_tr
    preds_tr.loc[val_mask, "p_pred"] = p_val_tr
    preds_tr.loc[test_mask, "p_pred"] = p_test_tr
    preds_tr["y_pred"] = (preds_tr["p_pred"] >= t_tr).astype(int)
    metrics_tr = {
        "train": _compute_metrics(y_train, p_train_tr, t_tr),
        "val": _compute_metrics(y_val, p_val_tr, t_tr),
        "test": _compute_metrics(y_test, p_test_tr, t_tr),
        "threshold": t_tr,
    }
    exp6_dir = out_base / "exp_exp6_transformer"
    exp6_dir.mkdir(parents=True, exist_ok=True)
    preds_tr.to_parquet(exp6_dir / "preds.parquet", index=False)
    with open(exp6_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_tr, f, indent=2)

    # EXP7 kNN analog
    knn_features = [
        *[f"dct_{i}" for i in range(30)],
        "slope_60m",
        "slope_120m",
        "drop_from_max",
        "plateau_frac_0p2",
    ]
    X_knn = df[knn_features]
    imputer_knn = SimpleImputer(strategy="median")
    X_knn_train = imputer_knn.fit_transform(X_knn[train_mask])
    X_knn_val = imputer_knn.transform(X_knn[val_mask])
    X_knn_test = imputer_knn.transform(X_knn[test_mask])

    mean_knn = X_knn_train.mean(axis=0, keepdims=True)
    std_knn = X_knn_train.std(axis=0, keepdims=True) + 1e-6
    X_knn_train = (X_knn_train - mean_knn) / std_knn
    X_knn_val = (X_knn_val - mean_knn) / std_knn
    X_knn_test = (X_knn_test - mean_knn) / std_knn

    best_k = None
    best_acc = -1.0
    best_val = None
    for k in [25, 50, 100, 200]:
        nn_model = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn_model.fit(X_knn_train)
        _, idx = nn_model.kneighbors(X_knn_val, return_distance=True)
        y_neighbors = y_train[idx]
        p_val_knn = y_neighbors.mean(axis=1)
        t_knn = _select_threshold(y_val, p_val_knn)
        acc = accuracy_score(y_val, p_val_knn >= t_knn)
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_val = p_val_knn

    nn_model = NearestNeighbors(n_neighbors=best_k, algorithm="auto")
    nn_model.fit(np.vstack([X_knn_train, X_knn_val]))
    _, idx_test = nn_model.kneighbors(X_knn_test, return_distance=True)
    y_trainval = np.concatenate([y_train, y_val])
    p_test_knn = y_trainval[idx_test].mean(axis=1)

    nn_train = NearestNeighbors(n_neighbors=best_k)
    nn_train.fit(X_knn_train)
    _, idx_train = nn_train.kneighbors(X_knn_train, return_distance=True)
    p_train_knn = y_train[idx_train].mean(axis=1)

    t_knn = _select_threshold(y_val, best_val)

    preds_knn = df[["target_date_local", "cutoff_utc", "y_hit_by_cutoff", "temp_max_sofar", "slope_60m", "drop_from_max", "split"]].copy()
    preds_knn.loc[train_mask, "p_pred"] = p_train_knn
    preds_knn.loc[val_mask, "p_pred"] = best_val
    preds_knn.loc[test_mask, "p_pred"] = p_test_knn
    preds_knn["y_pred"] = (preds_knn["p_pred"] >= t_knn).astype(int)
    metrics_knn = {
        "train": _compute_metrics(y_train, p_train_knn, t_knn),
        "val": _compute_metrics(y_val, best_val, t_knn),
        "test": _compute_metrics(y_test, p_test_knn, t_knn),
        "k": best_k,
        "threshold": t_knn,
    }
    exp7_dir = out_base / "exp_exp7_knn"
    exp7_dir.mkdir(parents=True, exist_ok=True)
    preds_knn.to_parquet(exp7_dir / "preds.parquet", index=False)
    with open(exp7_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_knn, f, indent=2)

    # Retrain base models on train+val for stacking test preds
    exp3_metrics = json.loads((out_base / "exp_exp3_lgbm" / "metrics.json").read_text(encoding="utf-8"))
    best_params = exp3_metrics.get("params")
    if best_params is None:
        raise RuntimeError("Missing best_params in exp3 metrics.json")

    trainval_mask = df["split"].isin(["train", "val"])
    X_trainval = imputer_main.transform(X[trainval_mask])
    y_trainval = y[trainval_mask.to_numpy()]
    best_iter = exp3_metrics.get("best_iteration")
    if not best_iter:
        best_iter = 200

    lgb_trainval = lgb.Dataset(X_trainval, label=y_trainval)
    lgbm_model_tv = lgb.train(best_params, lgb_trainval, num_boost_round=best_iter)
    p_test_lgbm_tv = lgbm_model_tv.predict(X_test)

    seq_trainval = np.concatenate([seq_train, seq_val], axis=0)
    y_trainval_seq = np.concatenate([y_train, y_val], axis=0)
    split_idx = int(len(seq_trainval) * 0.9)
    seq_tv_train = seq_trainval[:split_idx]
    seq_tv_val = seq_trainval[split_idx:]
    y_tv_train = y_trainval_seq[:split_idx]
    y_tv_val = y_trainval_seq[split_idx:]

    tv_train_loader = DataLoader(SequenceDataset(seq_tv_train, y_tv_train), batch_size=64, shuffle=True)
    tv_val_loader = DataLoader(SequenceDataset(seq_tv_val, y_tv_val), batch_size=128, shuffle=False)
    tcn_tv = TCN(n_features=seq_train.shape[2])
    tcn_tv = _train_torch_model(tcn_tv, tv_train_loader, tv_val_loader, pos_weight)
    p_test_tcn_tv = _predict_torch(tcn_tv, seq_test)

    # EXP8 stacking
    preds_rule = pd.read_parquet(out_base / "exp_exp1_rule" / "preds.parquet")
    preds_lgbm = pd.read_parquet(out_base / "exp_exp3_lgbm" / "preds.parquet")
    preds_tcn = pd.read_parquet(out_base / "exp_exp5_tcn" / "preds.parquet")

    base_val = pd.DataFrame({
        "exp1": preds_rule.loc[val_mask, "p_pred"].to_numpy(),
        "exp3": preds_lgbm.loc[val_mask, "p_pred"].to_numpy(),
        "exp5": preds_tcn.loc[val_mask, "p_pred"].to_numpy(),
        "exp7": preds_knn.loc[val_mask, "p_pred"].to_numpy(),
    })
    base_test = pd.DataFrame({
        "exp1": preds_rule.loc[test_mask, "p_pred"].to_numpy(),
        "exp3": p_test_lgbm_tv,
        "exp5": p_test_tcn_tv,
        "exp7": preds_knn.loc[test_mask, "p_pred"].to_numpy(),
    })
    base_train = pd.DataFrame({
        "exp1": preds_rule.loc[train_mask, "p_pred"].to_numpy(),
        "exp3": preds_lgbm.loc[train_mask, "p_pred"].to_numpy(),
        "exp5": preds_tcn.loc[train_mask, "p_pred"].to_numpy(),
        "exp7": preds_knn.loc[train_mask, "p_pred"].to_numpy(),
    })

    meta = LogisticRegression(max_iter=200)
    meta.fit(base_val, y_val)
    p_val_meta = meta.predict_proba(base_val)[:, 1]
    t_meta = _select_threshold(y_val, p_val_meta)

    p_train_meta = meta.predict_proba(base_train)[:, 1]
    p_test_meta = meta.predict_proba(base_test)[:, 1]

    preds_meta = df[["target_date_local", "cutoff_utc", "y_hit_by_cutoff", "temp_max_sofar", "slope_60m", "drop_from_max", "split"]].copy()
    preds_meta.loc[train_mask, "p_pred"] = p_train_meta
    preds_meta.loc[val_mask, "p_pred"] = p_val_meta
    preds_meta.loc[test_mask, "p_pred"] = p_test_meta
    preds_meta["y_pred"] = (preds_meta["p_pred"] >= t_meta).astype(int)

    metrics_meta = {
        "train": _compute_metrics(y_train, p_train_meta, t_meta),
        "val": _compute_metrics(y_val, p_val_meta, t_meta),
        "test": _compute_metrics(y_test, p_test_meta, t_meta),
        "threshold": t_meta,
    }
    exp8_dir = out_base / "exp_exp8_stack"
    exp8_dir.mkdir(parents=True, exist_ok=True)
    preds_meta.to_parquet(exp8_dir / "preds.parquet", index=False)
    with open(exp8_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_meta, f, indent=2)

    # Master report + metrics summary
    metrics_summary = {}
    for exp_name in ["exp1_rule", "exp2_seasonality", "exp3_lgbm", "exp4_regression", "exp5_tcn", "exp6_transformer", "exp7_knn", "exp8_stack"]:
        exp_dir = out_base / f"exp_{exp_name}"
        metrics_summary[exp_name] = json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8"))

    report_path = reports_dir / "tmax_hit1830_experiments_report.md"
    pos_rate = float(y_train.mean())
    always_no_acc = float(accuracy_score(y_test, np.zeros_like(y_test)))
    always_yes_acc = float(accuracy_score(y_test, np.ones_like(y_test)))

    metrics_table = []
    for name, metrics in metrics_summary.items():
        m = metrics["test"]
        metrics_table.append(
            {
                "exp": name,
                "test_acc": m["accuracy"],
                "test_bal_acc": m["balanced_accuracy"],
                "test_yes_recall": m["yes_recall"],
                "test_yes_precision": m["yes_precision"],
                "test_auc": m["roc_auc"],
            }
        )
    best_test = max(metrics_table, key=lambda r: r["test_acc"])

    def format_row(r: Dict[str, float]) -> str:
        return f"| {r['exp']} | {r['test_acc']:.3f} | {r['test_bal_acc']:.3f} | {r['test_yes_recall']:.3f} | {r['test_yes_precision']:.3f} | {r['test_auc']:.3f} |"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Tmax Hit by 18:30 Stockholm Experiments\n\n")
        f.write(f"Positive rate (train): {pos_rate:.3f}\n\n")
        f.write(f"Always-NO test accuracy: {always_no_acc:.3f}\n\n")
        f.write(f"Always-YES test accuracy: {always_yes_acc:.3f}\n\n")
        f.write("## Test Metrics Summary\n\n")
        f.write("| Experiment | Test Acc | Test Bal Acc | YES Recall | YES Precision | ROC AUC |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in metrics_table:
            f.write(format_row(row) + "\n")
        f.write("\n")
        f.write(f"Best test accuracy: {best_test['exp']} ({best_test['test_acc']:.3f})\n\n")

        best_name = best_test["exp"]
        preds_best = pd.read_parquet(out_base / f"exp_{best_name}" / "preds.parquet")
        preds_best = preds_best[preds_best["split"] == "test"].copy()
        preds_best["error_type"] = np.where(
            (preds_best["y_pred"] == 1) & (preds_best["y_hit_by_cutoff"] == 0),
            "FP",
            np.where((preds_best["y_pred"] == 0) & (preds_best["y_hit_by_cutoff"] == 1), "FN", "OK"),
        )
        fp = preds_best[preds_best["error_type"] == "FP"].sort_values("p_pred", ascending=False).head(20)
        fn = preds_best[preds_best["error_type"] == "FN"].sort_values("p_pred", ascending=True).head(20)

        f.write("## Error Analysis (Test, Best Model)\n\n")
        f.write("### False Positives (predicted YES, actually NO)\n\n")
        for _, r in fp.iterrows():
            f.write(
                f"- {r['target_date_local']} p={r['p_pred']:.3f} max_sofar={r['temp_max_sofar']:.1f} slope_60m={r['slope_60m']:.3f} drop_from_max={r['drop_from_max']:.2f}\n"
            )
        f.write("\n### False Negatives (predicted NO, actually YES)\n\n")
        for _, r in fn.iterrows():
            f.write(
                f"- {r['target_date_local']} p={r['p_pred']:.3f} max_sofar={r['temp_max_sofar']:.1f} slope_60m={r['slope_60m']:.3f} drop_from_max={r['drop_from_max']:.2f}\n"
            )

        f.write("\n## Acceptance Check\n\n")
        f.write(f"Best test accuracy: {best_test['test_acc']:.3f} (target 0.85)\n\n")
        f.write(f"Best test YES recall: {best_test['test_yes_recall']:.3f} (target 0.70)\n\n")

    with open(out_base / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
