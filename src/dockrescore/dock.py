"""Stage 2: AutoDock Vina docking through the python API; resume-safe.

Outputs per complex: poses.pdbqt (Vina), poses.sdf (Meeko-reconstructed RDKit molecules with
hydrogens), vina_scores.csv (rank, total/inter/intra/torsional terms).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from dockrescore.config import ComplexPaths, effective_cpu

SCORE_COLUMNS = ["vina_rank", "vina_total", "vina_inter", "vina_intra", "vina_torsion", "vina_intra_best"]


def pdbqt_poses_to_sdf(poses_pdbqt: Path, out_sdf: Path) -> int:
    """Convert Vina output PDBQT (multi-model) to an SDF with explicit hydrogens using Meeko."""
    from meeko import PDBQTMolecule, RDKitMolCreate

    pmol = PDBQTMolecule.from_file(str(poses_pdbqt), skip_typing=True)
    sdf_string, failures = RDKitMolCreate.write_sd_string(pmol)
    if failures:
        raise RuntimeError(f"Meeko could not rebuild poses {failures} from {poses_pdbqt}")
    out_sdf.write_text(sdf_string)
    return sdf_string.count("$$$$")


def _split_energy_row(e: list[float]) -> list[float]:
    """Map a ``Vina.energies()`` row to (total, inter, intra, torsion, intra_best).

    Vina 1.2 returns 5 columns [total, inter, intra, torsions, intra_best] for the vina/vinardo
    scoring functions, or the 8-column expanded form
    [total, lig_inter, flex_inter, other_inter, flex_intra, lig_intra, torsions, lig_intra_best].
    """
    if len(e) >= 8:
        return [e[0], e[1] + e[2] + e[3], e[4] + e[5], e[6], e[7]]
    if len(e) == 5:
        return list(e)
    return [*e, *([float("nan")] * (5 - len(e)))][:5]


def dock_complex(cp: ComplexPaths, cfg: dict[str, Any], force: bool = False) -> list[dict[str, float]]:
    """Dock one prepared complex with Vina; returns the score table (also written to CSV)."""
    if cp.scores_csv.exists() and cp.poses_sdf.exists() and not force:
        with cp.scores_csv.open() as fh:
            return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    from vina import Vina

    dcfg = cfg["dock"]
    box = json.loads(cp.box_json.read_text())
    v = Vina(sf_name=dcfg.get("scoring", "vina"), cpu=effective_cpu(cfg), seed=int(dcfg["seed"]), verbosity=0)
    v.set_receptor(str(cp.receptor_pdbqt))
    v.set_ligand_from_file(str(cp.ligand_pdbqt))
    v.compute_vina_maps(center=box["center"], box_size=box["size"])
    v.dock(exhaustiveness=int(dcfg["exhaustiveness"]), n_poses=int(dcfg["num_modes"]))
    n, er = int(dcfg["num_modes"]), float(dcfg["energy_range"])
    energies = v.energies(n_poses=n, energy_range=er)
    v.write_poses(str(cp.poses_pdbqt), n_poses=n, energy_range=er, overwrite=True)
    rows = []
    for i, e in enumerate(energies, start=1):
        rows.append(dict(zip(SCORE_COLUMNS, [i, *_split_energy_row([float(x) for x in e])])))
    with cp.scores_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SCORE_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    n_sdf = pdbqt_poses_to_sdf(cp.poses_pdbqt, cp.poses_sdf)
    if n_sdf != len(rows):
        raise RuntimeError(f"{cp.complex_id}: {n_sdf} SDF poses vs {len(rows)} energies")
    return rows
