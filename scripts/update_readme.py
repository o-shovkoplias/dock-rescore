#!/usr/bin/env python
"""Insert the metrics table + timings + versions into README.md between the RESULTS markers.

    python scripts/update_readme.py   (from the repo root, after `dockrescore report`)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

START, END = "<!-- RESULTS_TABLE_START -->", "<!-- RESULTS_TABLE_END -->"


def main() -> None:
    m = json.loads(Path("results/metrics.json").read_text())
    v = json.loads(Path("results/versions.json").read_text())
    t = pd.read_csv("results/timings.csv")
    pilot = m["n_complexes"] <= 10
    label = "Pilot run" if pilot else "Full run"
    lines = [
        f"{label}: **{m['n_complexes']} complexes, {m['n_poses']} poses, {m['n_features']} features**; "
        f"{m['cv_note']}. Near-native pose fraction {m['near_native_pose_fraction']:.2f}; "
        f"oracle top-1 (any pose < 2 Å) {m['oracle_top1_success']:.2f}.",
        "",
        "| metric | Vina rank | ML rescoring (out-of-fold) |",
        "|---|---|---|",
    ]
    for k in m["vina"]:
        f = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)  # noqa: E731
        lines.append(f"| {k} | {f(m['vina'][k])} | {f(m['ml'][k])} |")
    stage_cols = [c for c in t.columns if c not in ("complex_id", "total_s")]
    mean_total = t["total_s"].mean()
    if pilot:
        lines += ["", "Wall-clock per complex (s, 8 Vina threads on CPU):", "",
                  "| complex | " + " | ".join(stage_cols) + " | total |", "|---|" + "---|" * (len(stage_cols) + 1)]
        for _, r in t.iterrows():
            lines.append(f"| {r['complex_id']} | " + " | ".join(f"{r[c]:.1f}" for c in stage_cols) + f" | {r['total_s']:.1f} |")
        per_hour = 3600 / mean_total if mean_total > 0 else float("nan")
        lines += ["", f"Mean {mean_total:.0f} s per complex -> about {per_hour:.0f} complexes per CPU-hour "
                  f"(the 308-complex subset needs about {308*mean_total/3600:.1f} h at this exhaustiveness). Caveat: the "
                  f"pilot was drawn from small-to-medium ligands (15-35 heavy atoms) and receptors < 8000 atoms, so this "
                  f"is an optimistic lower bound; larger ligands/boxes dock slower."]
    else:
        lines += ["", f"Wall-clock over {len(t)} attempted complexes (s per complex, 8 Vina threads on CPU; full table in "
                  f"`results/timings.csv`): mean {mean_total:.1f}, median {t['total_s'].median():.1f}, max "
                  f"{t['total_s'].max():.1f}; of which docking mean {t['dock'].mean():.1f} s, features mean "
                  f"{t['features'].mean():.1f} s. Total {t['total_s'].sum()/3600:.2f} h (docking {t['dock'].sum()/3600:.2f} h)."]
    lines += ["", "Software: " + ", ".join(f"{k} {v[k]}" for k in ("vina", "meeko", "rdkit", "prolif", "openbabel", "torch", "MDAnalysis", "sklearn") if k in v) + "."]
    readme = Path("README.md").read_text()
    a, b = readme.index(START) + len(START), readme.index(END)
    Path("README.md").write_text(readme[:a] + "\n" + "\n".join(lines) + "\n" + readme[b:])
    print("[update_readme] README results block updated")


if __name__ == "__main__":
    main()
