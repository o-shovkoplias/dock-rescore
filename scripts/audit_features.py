#!/usr/bin/env python
"""Leakage / scope audit: retrain the GroupKFold MLP with suspicious feature blocks removed.

The docking box is centred on the crystal ligand, so ``geo_centroid_offset`` (pose centroid to
box centre) is a direct proxy for the translational part of the RMSD label. This script quantifies
how much of the reported ML gain rests on that feature. Writes results/audit/ablation.json and
results/audit/ablation.md; never touches results/metrics.json.

    python scripts/audit_features.py            (repo root, PYTHONPATH=src, "dock" env)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dockrescore.config import load_config
from dockrescore.features import META_COLUMNS
from dockrescore.model import cross_validate, ranking_metrics

SEEDS = (0, 1, 2)
VARIANTS: dict[str, callable] = {
    "full (as reported)": lambda c: True,
    "drop geo_centroid_offset": lambda c: c != "geo_centroid_offset",
    "drop geo_centroid_offset + cons_*": lambda c: c != "geo_centroid_offset" and not c.startswith("cons_"),
    "drop geo_centroid_offset + cons_* + plf_*": lambda c: c != "geo_centroid_offset" and not c.startswith(("cons_", "plf_")),
    "Vina terms only": lambda c: c.startswith("vina_"),
}


def per_complex_auc(df: pd.DataFrame, col: str, sign: float) -> float:
    from sklearn.metrics import roc_auc_score

    vals = []
    for _, g in df.groupby("complex_id"):
        if 0 < g["near_native"].sum() < len(g):
            vals.append(roc_auc_score(g["near_native"], sign * g[col]))
    return float(np.mean(vals))


def paired_bootstrap(df: pd.DataFrame, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of top-1(ML) - top-1(Vina) over complexes."""
    rng = np.random.default_rng(seed)
    ml, vina = [], []
    for _, g in df.groupby("complex_id"):
        ml.append(int(g.sort_values("ml_prob", ascending=False)["near_native"].iloc[0]))
        vina.append(int(g.sort_values("vina_total")["near_native"].iloc[0]))
    d = np.array(ml) - np.array(vina)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    cfg = load_config()
    feats = pd.read_csv("results/features.csv")
    out_dir = Path("results/audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    # heuristic baselines (no model)
    for name, col, hib in (("Vina rank", "vina_total", False), ("centroid offset alone (no model)", "geo_centroid_offset", False)):
        m = ranking_metrics(feats, col, higher_is_better=hib)
        m["per_complex_auc"] = per_complex_auc(feats, col, 1.0 if hib else -1.0)
        results[name] = {"n_features": 1, "seeds": {"-": m}}
    for name, keep in VARIANTS.items():
        cols = [c for c in feats.columns if c in META_COLUMNS or keep(c)]
        sub = feats[cols]
        per_seed = {}
        for s in SEEDS:
            cfg["model"]["seed"] = s
            preds, met = cross_validate(sub, cfg)
            m = met["ml"]
            m["per_complex_auc"] = per_complex_auc(preds, "ml_prob", 1.0)
            m["top1_delta_vs_vina_ci95"] = paired_bootstrap(preds, seed=s)
            per_seed[str(s)] = m
        results[name] = {"n_features": met["n_features"], "seeds": per_seed}
        print(name, {k: round(np.mean([v[k] for v in per_seed.values()]), 3) for k in ("top1_success", "roc_auc", "per_complex_auc")})
    (out_dir / "ablation.json").write_text(json.dumps(results, indent=2))
    lines = ["# Feature-ablation audit (GroupKFold(5) by complex, out-of-fold, 273 complexes / 4745 poses)", "",
             "Mean over MLP seeds 0,1,2 (min-max in brackets). `pooled AUC` is per-pose over all complexes",
             "(inflated by easy-vs-hard complex differences); `per-complex AUC` is the mean AUC within each",
             "complex that has both classes, which is what re-ranking actually needs.", "",
             "| variant | n_feat | top-1 success | pooled ROC-AUC | per-complex AUC | top-3 enrichment |", "|---|---|---|---|---|---|"]
    for name, r in results.items():
        ms = list(r["seeds"].values())
        def cell(k, fmt="{:.3f}"):
            v = [m[k] for m in ms]
            return fmt.format(np.mean(v)) + ("" if len(v) == 1 else f" [{min(v):.3f}-{max(v):.3f}]")
        lines.append(f"| {name} | {r['n_features']} | {cell('top1_success')} | {cell('roc_auc')} | {cell('per_complex_auc')} | {cell('top3_enrichment')} |")
    lines += ["", "95% paired-bootstrap CI (over complexes) of top-1(ML) - top-1(Vina), seed 0:"]
    for name, r in results.items():
        m = r["seeds"].get("0")
        if m and "top1_delta_vs_vina_ci95" in m:
            lo, hi = m["top1_delta_vs_vina_ci95"]
            lines.append(f"- {name}: [{lo:+.3f}, {hi:+.3f}]")
    (out_dir / "ablation.md").write_text("\n".join(lines) + "\n")
    print((out_dir / "ablation.md").read_text())


if __name__ == "__main__":
    main()
