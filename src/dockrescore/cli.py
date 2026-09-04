"""Typer CLI: dockrescore <stage> [--ids FILE] [--config FILE].

Stages: select-pilot, prepare, dock, rmsd, features, train, report, run-all, versions.
All stages are resume-safe: existing per-complex outputs are reused unless --force.
"""
from __future__ import annotations

import json
import random
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from dockrescore.config import Timer, complex_paths, load_config, read_ids, rpath, stage_done

app = typer.Typer(add_completion=False, help=__doc__)

CONFIG_OPT = typer.Option("config/config.yaml", "--config", help="YAML config (repo-relative).")
IDS_OPT = typer.Option(None, "--ids", help="Text file with complex ids (default: config pilot_ids).")
FORCE_OPT = typer.Option(False, "--force", help="Recompute even if outputs exist.")


def _log(msg: str) -> None:
    typer.echo(f"[dockrescore] {msg}")


@app.command()
def versions(config: str = CONFIG_OPT) -> None:
    """Print and save the versions of the chemistry/ML stack (results/versions.json)."""
    from dockrescore.versions import collect_versions

    cfg = load_config(config)
    v = collect_versions()
    out = rpath(cfg["paths"]["results_dir"]) / "versions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(v, indent=2))
    for k, val in v.items():
        _log(f"{k}: {val}")


@app.command("select-pilot")
def select_pilot(config: str = CONFIG_OPT) -> None:
    """Choose the pilot complexes deterministically from the 308 subset (ligand size window)."""
    cfg = load_config(config)
    p = cfg["pilot"]
    m = pd.read_csv(rpath(cfg["paths"]["manifest"]))
    pool = m[(m.in_308_subset == 1) & (m.ligand_heavy_atoms.between(p["min_heavy_atoms"], p["max_heavy_atoms"]))
             & (m.protein_atoms < 8000)]
    rng = random.Random(int(p["seed"]))
    ids = sorted(rng.sample(sorted(pool.complex_id), int(p["n_complexes"])))
    out = rpath(cfg["paths"]["pilot_ids"])
    out.write_text("# pilot complexes: 308-subset, %d-%d heavy atoms, <8000 protein atoms, seed %d\n" %
                   (p["min_heavy_atoms"], p["max_heavy_atoms"], p["seed"]) + "\n".join(ids) + "\n")
    _log(f"pilot pool {len(pool)} complexes -> {ids} written to {out}")


def _stage(name: str, fn, config: str, ids: Optional[str], force: bool) -> None:
    cfg = load_config(config)
    ok, failed = 0, []
    for cid in read_ids(cfg, ids):
        cp = complex_paths(cfg, cid)
        if not force and stage_done(cp, name):
            ok += 1
            _log(f"{name} {cid} cached, skipped (timing kept)")
            continue
        timer = Timer(cp.timing_json)
        try:
            with timer(name):
                fn(cp, cfg, force)
            ok += 1
            (cp.work_dir / f"{name}.error.txt").unlink(missing_ok=True)
            _log(f"{name} {cid} done in {timer.data[name]:.1f} s")
        except Exception as e:  # noqa: BLE001
            failed.append(cid)
            (cp.work_dir / f"{name}.error.txt").write_text(traceback.format_exc())
            _log(f"{name} {cid} FAILED: {type(e).__name__}: {e}")
    _log(f"{name}: {ok} ok, {len(failed)} failed {failed}")
    if failed and ok == 0:
        raise typer.Exit(code=1)


@app.command()
def prepare(config: str = CONFIG_OPT, ids: Optional[str] = IDS_OPT, force: bool = FORCE_OPT) -> None:
    """Stage 1: receptor/ligand PDBQT + docking box."""
    from dockrescore.prepare import prepare_complex

    _stage("prepare", prepare_complex, config, ids, force)


@app.command()
def dock(config: str = CONFIG_OPT, ids: Optional[str] = IDS_OPT, force: bool = FORCE_OPT) -> None:
    """Stage 2: Vina docking (python API), poses + scores."""
    from dockrescore.dock import dock_complex

    _stage("dock", dock_complex, config, ids, force)


@app.command()
def rmsd(config: str = CONFIG_OPT, ids: Optional[str] = IDS_OPT, force: bool = FORCE_OPT) -> None:
    """Stage 3: symmetry-corrected RMSD + near-native labels."""
    from dockrescore.rmsd import rmsd_complex

    _stage("rmsd", rmsd_complex, config, ids, force)


@app.command()
def features(config: str = CONFIG_OPT, ids: Optional[str] = IDS_OPT, force: bool = FORCE_OPT) -> None:
    """Stage 4: per-pose features -> results/features.csv (+ .parquet)."""
    from dockrescore.features import featurize_complex

    _stage("features", featurize_complex, config, ids, force)
    cfg = load_config(config)
    parts = []
    for cid in read_ids(cfg, ids):
        f = complex_paths(cfg, cid).features_csv
        if f.exists():
            parts.append(pd.read_csv(f))
    df = pd.concat(parts, ignore_index=True)
    out = rpath(cfg["paths"]["results_dir"]) / "features.csv"
    df.to_csv(out, index=False)
    try:
        df.to_parquet(out.with_suffix(".parquet"), index=False)
    except Exception as e:  # noqa: BLE001
        _log(f"parquet export skipped: {e}")
    _log(f"features table: {df.shape} -> {out}")


@app.command()
def train(config: str = CONFIG_OPT) -> None:
    """Stage 5: GroupKFold MLP; results/metrics.json + predictions.csv."""
    from dockrescore.model import cross_validate, save_outputs

    cfg = load_config(config)
    rd = rpath(cfg["paths"]["results_dir"])
    feats = pd.read_csv(rd / "features.csv")
    preds, metrics = cross_validate(feats, cfg)
    save_outputs(preds, metrics, rd)
    _log(json.dumps(metrics, indent=2))


@app.command()
def report(config: str = CONFIG_OPT, ids: Optional[str] = IDS_OPT) -> None:
    """Stage 6: figures/top1_success.png, figures/rmsd_vs_rank.png, results/timings.csv, results/report.md."""
    from dockrescore.report import plot_rmsd_vs_rank, plot_top1, timing_table, write_report
    from dockrescore.versions import collect_versions

    cfg = load_config(config)
    rd, fd = rpath(cfg["paths"]["results_dir"]), rpath(cfg["paths"]["figures_dir"])
    metrics = json.loads((rd / "metrics.json").read_text())
    preds = pd.read_csv(rd / "predictions.csv")
    plot_top1(metrics, fd / "top1_success.png")
    plot_rmsd_vs_rank(preds, fd / "rmsd_vs_rank.png")
    timings = timing_table(rpath(cfg["paths"]["work_dir"]), read_ids(cfg, ids))
    timings.to_csv(rd / "timings.csv", index=False)
    write_report(metrics, timings, collect_versions(), rd / "report.md")
    _log(f"figures -> {fd}, report -> {rd / 'report.md'}")


@app.command("run-all")
def run_all(config: str = CONFIG_OPT, ids: Optional[str] = IDS_OPT, force: bool = FORCE_OPT) -> None:
    """Stages 1-6 in sequence."""
    versions(config)
    prepare(config, ids, force)
    dock(config, ids, force)
    rmsd(config, ids, force)
    features(config, ids, force)
    train(config)
    report(config, ids)


if __name__ == "__main__":
    app()
