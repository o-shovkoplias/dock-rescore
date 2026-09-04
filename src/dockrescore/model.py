"""Stage 5: PyTorch MLP rescoring model with GroupKFold cross-validation by complex."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn

from dockrescore.features import feature_columns


class MLP(nn.Module):
    """Two hidden layers, ReLU, dropout, single logit output (BCE-with-logits loss)."""

    def __init__(self, n_in: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x).squeeze(-1)


def _fit(x: np.ndarray, y: np.ndarray, mcfg: dict[str, Any], seed: int) -> MLP:
    torch.set_num_threads(int(mcfg.get("torch_threads", 4)))  # tiny batches: more threads only add spin overhead
    torch.manual_seed(seed)
    dev = torch.device(mcfg.get("device", "cpu"))
    model = MLP(x.shape[1], list(mcfg["hidden"]), float(mcfg["dropout"])).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(mcfg["lr"]), weight_decay=float(mcfg["weight_decay"]))
    pos = max(y.sum(), 1.0)
    pos_weight = torch.tensor([(len(y) - pos) / pos], dtype=torch.float32, device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    xt, yt = torch.tensor(x, dtype=torch.float32, device=dev), torch.tensor(y, dtype=torch.float32, device=dev)
    bs = int(mcfg["batch_size"])
    model.train()
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(int(mcfg["epochs"])):
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    return model


def _predict(model: MLP, x: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(x, dtype=torch.float32, device=torch.device(device)))).cpu().numpy()


def ranking_metrics(df: pd.DataFrame, score_col: str, higher_is_better: bool, top_k: int = 3) -> dict[str, float]:
    """Top-1 success, top-k near-native enrichment and per-pose AUC for one ranking column.

    top1_success: fraction of complexes whose best-ranked pose has RMSD < threshold.
    topk_enrichment: (near-native fraction among the top-k poses) / (near-native fraction overall).
    """
    succ, topk_hits, topk_n = [], 0, 0
    for _, g in df.groupby("complex_id"):
        order = g.sort_values(score_col, ascending=not higher_is_better)
        succ.append(int(order["near_native"].iloc[0]))
        topk_hits += int(order["near_native"].iloc[:top_k].sum())
        topk_n += min(top_k, len(order))
    base_rate = df["near_native"].mean()
    out = {
        "top1_success": float(np.mean(succ)),
        "top1_hits": int(np.sum(succ)),
        f"top{top_k}_near_native_fraction": topk_hits / max(topk_n, 1),
        f"top{top_k}_enrichment": (topk_hits / max(topk_n, 1)) / base_rate if base_rate > 0 else float("nan"),
    }
    y = df["near_native"].values
    if 0 < y.sum() < len(y):
        s = df[score_col].values if higher_is_better else -df[score_col].values
        out["roc_auc"] = float(roc_auc_score(y, s))
    else:
        out["roc_auc"] = float("nan")
    return out


def cross_validate(features: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """GroupKFold (by complex) out-of-fold predictions and Vina-vs-ML metrics."""
    mcfg = cfg["model"]
    cols = feature_columns(features)
    df = features.copy()
    x_all = df[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y_all = df["near_native"].values.astype(float)
    groups = df["complex_id"].values
    n_groups = len(np.unique(groups))
    n_folds = min(int(mcfg["n_folds"]), n_groups)
    df["ml_prob"] = np.nan
    if n_folds < 2 or y_all.sum() == 0 or y_all.sum() == len(y_all):
        note = "not enough complexes / label variety for cross-validation; ML predictions skipped"
        df["ml_prob"] = 0.0
    else:
        note = f"GroupKFold n_folds={n_folds} over {n_groups} complexes"
        for k, (tr, te) in enumerate(GroupKFold(n_splits=n_folds).split(x_all, y_all, groups)):
            mu, sd = x_all[tr].mean(0), x_all[tr].std(0) + 1e-8
            xtr, xte = (x_all[tr] - mu) / sd, (x_all[te] - mu) / sd
            if y_all[tr].sum() == 0 or y_all[tr].sum() == len(tr):
                df.loc[df.index[te], "ml_prob"] = 0.5  # degenerate fold: no label variety in training data
                continue
            model = _fit(xtr, y_all[tr], mcfg, int(mcfg["seed"]) + k)
            df.loc[df.index[te], "ml_prob"] = _predict(model, xte, mcfg.get("device", "cpu"))
    n_cplx = df["complex_id"].nunique()
    metrics = {
        "n_complexes": int(n_cplx),
        "n_poses": int(len(df)),
        "near_native_pose_fraction": float(y_all.mean()),
        "oracle_top1_success": float(df.groupby("complex_id")["near_native"].max().mean()),
        "vina": ranking_metrics(df, "vina_total", higher_is_better=False),
        "ml": ranking_metrics(df, "ml_prob", higher_is_better=True),
        "n_features": len(cols),
        "cv_note": note,
    }
    return df, metrics


def save_outputs(preds: pd.DataFrame, metrics: dict[str, Any], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(results_dir / "predictions.csv", index=False)
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
