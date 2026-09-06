"""Stage 6: figures and a short markdown report from results/metrics.json and timings."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical slots from the dataviz reference palette (fixed order, not cycled).
C_VINA, C_ML, C_ORACLE = "#2a78d6", "#eb6834", "#9a9a95"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


def plot_top1(metrics: dict[str, Any], out: Path) -> None:
    """Bar chart: top-1 success rate of Vina ranking vs ML rescoring vs oracle (any pose < 2 A)."""
    labels = ["Vina rank", "ML rescoring", "Oracle\n(any pose < 2 Å)"]
    vals = [metrics["vina"]["top1_success"], metrics["ml"]["top1_success"], metrics["oracle_top1_success"]]
    n = metrics["n_complexes"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bars = ax.bar(labels, vals, color=[C_VINA, C_ML, C_ORACLE], width=0.55, linewidth=0)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", va="bottom", color=INK, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Top-1 success rate (RMSD < 2 Å)", color=INK2)
    ax.set_title(f"Redocking top-1 success, n = {n} complexes" + (" (PILOT)" if n <= 10 else ""), color=INK, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d9d8d3")
    ax.tick_params(colors=INK2)
    ax.yaxis.grid(True, color="#e8e7e2", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_rmsd_vs_rank(preds: pd.DataFrame, out: Path) -> None:
    """Strip plot of pose RMSD vs Vina rank per complex, near-native band shaded."""
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhspan(0, 2.0, color="#e3eefb", zorder=0)
    n = preds["complex_id"].nunique()
    small = n <= 10  # per-complex legend only makes sense for a pilot-sized run
    for i, (cid, g) in enumerate(preds.groupby("complex_id")):
        ax.plot(g["vina_rank"], g["rmsd"], "o-", ms=3.5, lw=1.2, color=C_VINA,
                alpha=(0.35 + 0.65 * (i == 0)) if small else 0.15, label=cid if small else None)
    ax.set_xlabel("Vina rank", color=INK2)
    ax.set_ylabel("Heavy-atom RMSD to crystal (Å)", color=INK2)
    ax.set_title(f"Pose RMSD by Vina rank, n = {n} complexes (shaded: near-native < 2 Å)", color=INK, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if small:
        ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def timing_table(work_dir: Path, ids: list[str]) -> pd.DataFrame:
    rows = []
    for cid in ids:
        f = work_dir / cid / "timing.json"
        if f.exists():
            d = json.loads(f.read_text())
            d["complex_id"] = cid
            rows.append(d)
    df = pd.DataFrame(rows)
    if len(df):
        stage_cols = [c for c in df.columns if c != "complex_id"]
        df["total_s"] = df[stage_cols].sum(axis=1)
        df = df[["complex_id", *stage_cols, "total_s"]]
    return df


def write_report(metrics: dict[str, Any], timings: pd.DataFrame, versions: dict[str, Any], out: Path) -> None:
    lines = ["# dock-rescore report", "", f"Complexes: {metrics['n_complexes']}, poses: {metrics['n_poses']}, "
             f"features: {metrics['n_features']}", f"CV: {metrics['cv_note']}", "",
             "| metric | Vina rank | ML rescoring |", "|---|---|---|"]
    for k in metrics["vina"]:
        v1, v2 = metrics["vina"][k], metrics["ml"][k]
        fmt = (lambda x: f"{x:.3f}" if isinstance(x, float) else str(x))
        lines.append(f"| {k} | {fmt(v1)} | {fmt(v2)} |")
    lines += ["", f"Oracle top-1 (any near-native pose): {metrics['oracle_top1_success']:.3f}",
              f"Near-native pose fraction: {metrics['near_native_pose_fraction']:.3f}", "", "## Timing per complex (s)", ""]
    if len(timings):
        lines.append(timings.to_markdown(index=False, floatfmt=".1f"))
        lines += ["", f"Mean total per complex: {timings['total_s'].mean():.1f} s"]
    lines += ["", "## Software versions", ""] + [f"- {k}: {v}" for k, v in versions.items()]
    out.write_text("\n".join(lines) + "\n")
