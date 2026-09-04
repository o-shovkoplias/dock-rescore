"""Configuration loading and repo-relative path helpers."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load the YAML config; relative paths are interpreted from the repo root."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return yaml.safe_load(p.read_text())


def rpath(rel: str | Path) -> Path:
    """Resolve a repo-relative path to an absolute one."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def effective_cpu(cfg: dict[str, Any]) -> int:
    """CPU threads for Vina: min(config, max_cpu_fraction * nproc), at least 1."""
    n = os.cpu_count() or 1
    frac = float(cfg["dock"].get("max_cpu_fraction", 0.5))
    return max(1, min(int(cfg["dock"]["cpu"]), int(n * frac)))


@dataclass
class ComplexPaths:
    """All files belonging to one PoseBusters complex, raw and derived."""

    complex_id: str
    raw_dir: Path
    work_dir: Path

    @property
    def protein_pdb(self) -> Path:
        return self.raw_dir / f"{self.complex_id}_protein.pdb"

    @property
    def ligand_sdf(self) -> Path:
        return self.raw_dir / f"{self.complex_id}_ligand.sdf"

    @property
    def receptor_pdbqt(self) -> Path:
        return self.work_dir / "receptor.pdbqt"

    @property
    def receptor_h_pdb(self) -> Path:
        return self.work_dir / "receptor_h.pdb"

    @property
    def ligand_pdbqt(self) -> Path:
        return self.work_dir / "ligand.pdbqt"

    @property
    def box_json(self) -> Path:
        return self.work_dir / "box.json"

    @property
    def poses_pdbqt(self) -> Path:
        return self.work_dir / "poses.pdbqt"

    @property
    def poses_sdf(self) -> Path:
        return self.work_dir / "poses.sdf"

    @property
    def scores_csv(self) -> Path:
        return self.work_dir / "vina_scores.csv"

    @property
    def rmsd_csv(self) -> Path:
        return self.work_dir / "rmsd.csv"

    @property
    def features_csv(self) -> Path:
        return self.work_dir / "features.csv"

    @property
    def timing_json(self) -> Path:
        return self.work_dir / "timing.json"


def stage_done(cp: ComplexPaths, stage: str) -> bool:
    """True if the outputs of ``stage`` already exist for this complex (used to skip cached work)."""
    outputs = {
        "prepare": (cp.receptor_pdbqt, cp.receptor_h_pdb, cp.ligand_pdbqt, cp.box_json),
        "dock": (cp.poses_pdbqt, cp.poses_sdf, cp.scores_csv),
        "rmsd": (cp.rmsd_csv,),
        "features": (cp.features_csv,),
    }[stage]
    return all(p.exists() for p in outputs)


def complex_paths(cfg: dict[str, Any], complex_id: str) -> ComplexPaths:
    cp = ComplexPaths(
        complex_id=complex_id,
        raw_dir=rpath(cfg["paths"]["raw_dir"]) / complex_id,
        work_dir=rpath(cfg["paths"]["work_dir"]) / complex_id,
    )
    cp.work_dir.mkdir(parents=True, exist_ok=True)
    return cp


def read_ids(cfg: dict[str, Any], ids_file: str | None = None) -> list[str]:
    """Complex ids to process: explicit file, else the pilot list from the config."""
    f = rpath(ids_file or cfg["paths"]["pilot_ids"])
    ids = []
    for line in f.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line.split()[0])
    return ids


class Timer:
    """Append wall-clock timings for named stages to a per-complex timing.json."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, float] = json.loads(path.read_text()) if path.exists() else {}

    def __call__(self, stage: str) -> "_Stage":
        return _Stage(self, stage)

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))


@dataclass
class _Stage:
    timer: Timer
    stage: str
    t0: float = field(default=0.0)

    def __enter__(self) -> "_Stage":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.timer.data[self.stage] = round(time.perf_counter() - self.t0, 3)
        self.timer.save()
