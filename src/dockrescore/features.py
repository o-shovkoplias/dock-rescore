"""Stage 4: per-pose feature table.

Three feature blocks per pose:
  * ligand 2D descriptors (RDKit, fixed list of 40; identical for all poses of a complex),
  * ProLIF interaction-fingerprint counts (protein pocket within ``pocket_cutoff`` of the crystal
    ligand, protonated receptor from stage 1),
  * geometric / docking features: Vina terms and rank, heavy-atom contacts, buried fraction,
    min protein distance, box-centre offset, and pose-consensus (number of other poses within 2 A).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from dockrescore.config import ComplexPaths
from dockrescore.rmsd import pairwise_pose_rmsd

RDLogger.DisableLog("rdApp.*")

# vdW radii (A) missing from MDAnalysis' default bond-guessing table (halogens, metals, both cases).
EXTRA_VDW: dict[str, float] = {
    k: v
    for base, v in {"CL": 1.75, "BR": 1.85, "I": 1.98, "F": 1.47, "ZN": 1.39, "MG": 1.73, "CA": 1.90, "NA": 2.27,
                    "K": 2.75, "FE": 1.50, "MN": 1.60, "CO": 1.50, "NI": 1.60, "CU": 1.40, "SE": 1.90, "CD": 1.58,
                    "HG": 1.55, "MO": 2.10, "W": 2.10}.items()
    for k in (base, base.capitalize())
}

# Fixed list of 40 RDKit 2D descriptors (name -> callable(mol)).
RDKIT_2D: dict[str, Any] = {
    "MolWt": Descriptors.MolWt,
    "HeavyAtomCount": Lipinski.HeavyAtomCount,
    "NumHDonors": Lipinski.NumHDonors,
    "NumHAcceptors": Lipinski.NumHAcceptors,
    "NumRotatableBonds": Lipinski.NumRotatableBonds,
    "RingCount": Lipinski.RingCount,
    "NumAromaticRings": Lipinski.NumAromaticRings,
    "NumAliphaticRings": Lipinski.NumAliphaticRings,
    "NumSaturatedRings": Lipinski.NumSaturatedRings,
    "NumHeteroatoms": Lipinski.NumHeteroatoms,
    "FractionCSP3": Lipinski.FractionCSP3,
    "NHOHCount": Lipinski.NHOHCount,
    "NOCount": Lipinski.NOCount,
    "MolLogP": Crippen.MolLogP,
    "MolMR": Crippen.MolMR,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "LabuteASA": rdMolDescriptors.CalcLabuteASA,
    "NumAmideBonds": rdMolDescriptors.CalcNumAmideBonds,
    "NumSpiroAtoms": rdMolDescriptors.CalcNumSpiroAtoms,
    "NumBridgeheadAtoms": rdMolDescriptors.CalcNumBridgeheadAtoms,
    "NumAtomStereoCenters": rdMolDescriptors.CalcNumAtomStereoCenters,
    "Chi0v": rdMolDescriptors.CalcChi0v,
    "Chi1v": rdMolDescriptors.CalcChi1v,
    "Chi2v": rdMolDescriptors.CalcChi2v,
    "Kappa1": rdMolDescriptors.CalcKappa1,
    "Kappa2": rdMolDescriptors.CalcKappa2,
    "Kappa3": rdMolDescriptors.CalcKappa3,
    "HallKierAlpha": rdMolDescriptors.CalcHallKierAlpha,
    "BalabanJ": Descriptors.BalabanJ,
    "BertzCT": Descriptors.BertzCT,
    "MaxPartialCharge": Descriptors.MaxPartialCharge,
    "MinPartialCharge": Descriptors.MinPartialCharge,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "NumRadicalElectrons": Descriptors.NumRadicalElectrons,
    "FormalCharge": lambda m: Chem.GetFormalCharge(m),
    "NumHalogens": lambda m: sum(1 for a in m.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I")),
    "NumSulfur": lambda m: sum(1 for a in m.GetAtoms() if a.GetSymbol() == "S"),
    "NumPhosphorus": lambda m: sum(1 for a in m.GetAtoms() if a.GetSymbol() == "P"),
    "fr_COO": Descriptors.fr_COO,
    "fr_NH2": Descriptors.fr_NH2,
}
assert len(RDKIT_2D) == 40


def ligand_descriptors(mol: Chem.Mol) -> dict[str, float]:
    m = Chem.RemoveHs(mol)
    out = {}
    for k, f in RDKIT_2D.items():
        try:
            out[f"lig_{k}"] = float(f(m))
        except Exception:  # noqa: BLE001
            out[f"lig_{k}"] = float("nan")
    return out


def _protein_heavy_xyz(pdb: Path) -> np.ndarray:
    xyz = []
    with pdb.open() as fh:
        for ln in fh:
            if not ln.startswith(("ATOM", "HETATM")):
                continue
            elem, name = ln[76:78].strip(), ln[12:16].strip()
            if elem == "H" or (not elem and name[:1] == "H"):
                continue
            xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    return np.asarray(xyz)


def geometric_features(pose: Chem.Mol, prot_xyz: np.ndarray, center: np.ndarray, cfg: dict[str, Any]) -> dict[str, float]:
    heavy = [a.GetIdx() for a in pose.GetAtoms() if a.GetAtomicNum() > 1]
    lxyz = pose.GetConformer().GetPositions()[heavy]
    d = np.linalg.norm(lxyz[:, None, :] - prot_xyz[None, :, :], axis=-1)
    cc, bc = float(cfg["features"]["contact_cutoff"]), float(cfg["features"]["buried_cutoff"])
    dmin = d.min(axis=1)
    return {
        "geo_n_contacts": float((d < cc).sum()),
        "geo_n_contacts_per_heavy": float((d < cc).sum() / len(heavy)),
        "geo_buried_fraction": float((dmin < bc).mean()),
        "geo_min_dist": float(dmin.min()),
        "geo_mean_min_dist": float(dmin.mean()),
        "geo_n_close_lt_2p5": float((d < 2.5).sum()),
        "geo_centroid_offset": float(np.linalg.norm(lxyz.mean(0) - center)),
        "geo_radius_gyration": float(np.sqrt(((lxyz - lxyz.mean(0)) ** 2).sum(1).mean())),
    }


def prolif_counts(poses: list[Chem.Mol], cp: ComplexPaths, cfg: dict[str, Any]) -> pd.DataFrame:
    """Per-pose counts of each ProLIF interaction type (summed over residues)."""
    import MDAnalysis as mda
    import prolif as plf

    inter = list(cfg["features"]["prolif_interactions"])
    cut = float(cfg["features"]["pocket_cutoff"])
    box = json.loads(cp.box_json.read_text())
    cx, cy, cz = box["center"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(str(cp.receptor_h_pdb))
        half = max(box["size"]) / 2 + cut
        sel = u.select_atoms(f"byres (point {cx} {cy} {cz} {half})")
        sel = sel.select_atoms("not resname HOH WAT")
        sel.guess_bonds(vdwradii=EXTRA_VDW)
        prot = plf.Molecule.from_mda(sel, NoImplicit=False)
        fp = plf.Fingerprint(inter, count=True)
        ligs = [plf.Molecule.from_rdkit(m) for m in poses]
        fp.run_from_iterable(ligs, prot, progress=False, n_jobs=1)
        df = fp.to_dataframe()
    out = pd.DataFrame(0.0, index=range(len(poses)), columns=[f"plf_{i}" for i in inter])
    if len(df):
        for (_, _, itype), col in df.T.iterrows():
            out.loc[col.index, f"plf_{itype}"] += col.values.astype(float)
    out["plf_total"] = out.sum(axis=1)
    return out


def featurize_complex(cp: ComplexPaths, cfg: dict[str, Any], force: bool = False) -> pd.DataFrame:
    """Build the pose feature table for one complex (features.csv)."""
    if cp.features_csv.exists() and not force:
        return pd.read_csv(cp.features_csv)
    poses = [m for m in Chem.SDMolSupplier(str(cp.poses_sdf), removeHs=False) if m is not None]
    scores = pd.read_csv(cp.scores_csv)
    rmsd = pd.read_csv(cp.rmsd_csv)
    assert len(poses) == len(scores) == len(rmsd), cp.complex_id
    box = json.loads(cp.box_json.read_text())
    prot_xyz = _protein_heavy_xyz(cp.protein_pdb)
    lig2d = ligand_descriptors(poses[0])
    geo = pd.DataFrame([geometric_features(m, prot_xyz, np.array(box["center"]), cfg) for m in poses])
    thr = float(cfg["rmsd"]["near_native_threshold"])
    pw = pairwise_pose_rmsd(poses)
    consensus = pd.DataFrame(
        {
            "cons_n_within_2A": (pw < thr).sum(axis=1) - 1,
            "cons_mean_rmsd_to_others": np.where(len(poses) > 1, pw.sum(axis=1) / max(len(poses) - 1, 1), 0.0),
            "cons_rmsd_to_top1": pw[0],
        }
    )
    plf_df = prolif_counts(poses, cp, cfg)
    df = pd.concat([scores.reset_index(drop=True), rmsd[["rmsd", "near_native"]], geo, consensus, plf_df], axis=1)
    for k, v in lig2d.items():
        df[k] = v
    df.insert(0, "complex_id", cp.complex_id)
    df["vina_rank_frac"] = (df["vina_rank"] - 1) / max(len(df) - 1, 1)
    df["vina_delta_to_best"] = df["vina_total"] - df["vina_total"].min()
    df["vina_total_per_heavy"] = df["vina_total"] / box["n_ligand_heavy"]
    df.to_csv(cp.features_csv, index=False)
    return df


META_COLUMNS = ["complex_id", "vina_rank", "rmsd", "near_native"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns: everything except identifiers and labels."""
    return [c for c in df.columns if c not in META_COLUMNS]
