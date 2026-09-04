"""Stage 3: symmetry-corrected heavy-atom RMSD of each pose against the crystal ligand."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from dockrescore.config import ComplexPaths


def _heavy(mol: Chem.Mol) -> Chem.Mol:
    m = Chem.RemoveHs(mol, sanitize=False)
    m.UpdatePropertyCache(strict=False)
    return m


def symm_rmsd(pose: Chem.Mol, ref: Chem.Mol) -> float:
    """Symmetry-aware RMSD *without* re-alignment (poses are already in the receptor frame).

    Uses RDKit ``CalcRMS`` (enumerates graph automorphisms). If the two graphs do not match
    (e.g. differing bond orders after PDBQT round-trip), falls back to a substructure match
    that ignores bond orders and to a plain element-ordered RMSD as last resort.
    """
    p, r = _heavy(pose), _heavy(ref)
    try:
        return float(rdMolAlign.CalcRMS(p, r))
    except RuntimeError:
        pass
    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    query = Chem.AdjustQueryProperties(r, params)
    matches = p.GetSubstructMatches(query, uniquify=False, useChirality=False, maxMatches=10000)
    pxyz, rxyz = p.GetConformer().GetPositions(), r.GetConformer().GetPositions()
    if not matches:
        if p.GetNumAtoms() != r.GetNumAtoms():
            return float("nan")
        return float(np.sqrt(((pxyz - rxyz) ** 2).sum(1).mean()))
    best = min(np.sqrt(((pxyz[list(m)] - rxyz) ** 2).sum(1).mean()) for m in matches)
    return float(best)


def rmsd_complex(cp: ComplexPaths, cfg: dict[str, Any], force: bool = False) -> list[dict[str, float]]:
    """RMSD of every pose in poses.sdf vs the crystal ligand; writes rmsd.csv."""
    if cp.rmsd_csv.exists() and not force:
        with cp.rmsd_csv.open() as fh:
            return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    thr = float(cfg["rmsd"]["near_native_threshold"])
    ref = Chem.MolFromMolFile(str(cp.ligand_sdf), removeHs=False)
    poses = [m for m in Chem.SDMolSupplier(str(cp.poses_sdf), removeHs=False, sanitize=False) if m is not None]
    rows = []
    for i, m in enumerate(poses, start=1):
        r = symm_rmsd(m, ref)
        rows.append({"vina_rank": i, "rmsd": round(r, 4), "near_native": int(r < thr)})
    with cp.rmsd_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["vina_rank", "rmsd", "near_native"])
        w.writeheader()
        w.writerows(rows)
    return rows


def pairwise_pose_rmsd(poses: list[Chem.Mol]) -> np.ndarray:
    """Symmetric matrix of symmetry-corrected RMSDs between poses (for consensus features)."""
    n = len(poses)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = symm_rmsd(poses[i], poses[j])
    return d
