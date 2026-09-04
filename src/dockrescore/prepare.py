"""Stage 1: receptor/ligand PDBQT preparation and docking-box definition.

Receptor: OpenBabel (protonate, write rigid PDBQT; non-polar H merged) or Meeko.
Ligand:   Meeko from the crystal SDF (protonation state as given in the file, explicit H added
          only where RDKit's valence model needs them).
Box:      crystal-ligand centroid, edge = ligand extent + 2 * padding, floored at box_min_size.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from dockrescore.config import ComplexPaths


def load_crystal_ligand(sdf: Path) -> Chem.Mol:
    """Crystal ligand with explicit hydrogens (coordinates for added H come from RDKit)."""
    mol = Chem.MolFromMolFile(str(sdf), removeHs=False)
    if mol is None:
        raise ValueError(f"RDKit could not parse {sdf}")
    return Chem.AddHs(mol, addCoords=True)


def docking_box(mol: Chem.Mol, padding: float, min_size: float) -> dict[str, list[float]]:
    """Axis-aligned box around the heavy atoms of ``mol``."""
    conf = mol.GetConformer()
    xyz = np.array([list(conf.GetAtomPosition(a.GetIdx())) for a in mol.GetAtoms() if a.GetAtomicNum() > 1])
    center = xyz.mean(axis=0)
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    size = np.maximum(extent + 2.0 * padding, min_size)
    return {"center": [round(float(x), 3) for x in center], "size": [round(float(x), 3) for x in size]}


def ligand_to_pdbqt(mol: Chem.Mol) -> str:
    """Ligand PDBQT string via Meeko (version-tolerant for meeko 0.5/0.6)."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    prep = MoleculePreparation()
    setups = prep.prepare(mol)
    setup = setups[0] if isinstance(setups, (list, tuple)) else prep.setup
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setup)
    if not ok:
        raise RuntimeError(f"Meeko failed: {err}")
    return pdbqt


def receptor_to_pdbqt_openbabel(pdb: Path, out_pdbqt: Path, out_h_pdb: Path, add_h: bool = True) -> None:
    """Protonate the receptor with OpenBabel and write a rigid PDBQT plus a protonated PDB.

    Waters are dropped (PoseBusters files are already solvent-free); cofactors and metals kept.
    """
    from openbabel import openbabel as ob
    from openbabel import pybel

    ob.obErrorLog.SetOutputLevel(0)
    mol = next(pybel.readfile("pdb", str(pdb)))
    # remove any residual waters
    to_del = [a.OBAtom for a in mol.atoms if a.OBAtom.GetResidue() and a.OBAtom.GetResidue().GetName() in ("HOH", "WAT")]
    mol.OBMol.BeginModify()
    for a in to_del:
        mol.OBMol.DeleteAtom(a)
    mol.OBMol.EndModify()
    if add_h:
        mol.OBMol.AddHydrogens(False, True, 7.4)  # polaronly=False, correctForPH=True
    mol.write("pdb", str(out_h_pdb), overwrite=True)
    mol.write("pdbqt", str(out_pdbqt), overwrite=True, opt={"r": None})  # -xr : rigid receptor, no ROOT/BRANCH


def receptor_to_pdbqt_meeko(pdb: Path, out_pdbqt: Path, out_h_pdb: Path) -> None:
    """Receptor PDBQT via Meeko's Polymer API (meeko >= 0.6); strict about residue templates."""
    from meeko import PDBQTWriterLegacy, Polymer, ResidueChemTemplates

    templates = ResidueChemTemplates.create_from_defaults()
    from meeko import MoleculePreparation

    pol = Polymer.from_pdb_string(pdb.read_text(), templates, MoleculePreparation(), allow_bad_res=True)
    pdbqt_str = PDBQTWriterLegacy.write_string_from_polymer(pol)
    out_pdbqt.write_text(pdbqt_str[0] if isinstance(pdbqt_str, tuple) else pdbqt_str)
    out_h_pdb.write_text(pol.to_pdb())


def prepare_complex(cp: ComplexPaths, cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Run receptor + ligand preparation for one complex; resume-safe."""
    pcfg = cfg["prepare"]
    done = all(p.exists() for p in (cp.receptor_pdbqt, cp.ligand_pdbqt, cp.box_json, cp.receptor_h_pdb))
    if done and not force:
        return json.loads(cp.box_json.read_text())
    mol = load_crystal_ligand(cp.ligand_sdf)
    box = docking_box(mol, float(pcfg["box_padding"]), float(pcfg["box_min_size"]))
    cp.ligand_pdbqt.write_text(ligand_to_pdbqt(mol))
    if pcfg.get("receptor_method", "openbabel") == "meeko":
        receptor_to_pdbqt_meeko(cp.protein_pdb, cp.receptor_pdbqt, cp.receptor_h_pdb)
    else:
        receptor_to_pdbqt_openbabel(cp.protein_pdb, cp.receptor_pdbqt, cp.receptor_h_pdb, bool(pcfg["add_hydrogens"]))
    box["n_ligand_heavy"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    box["n_rotatable_bonds"] = int(AllChem.CalcNumRotatableBonds(Chem.RemoveHs(mol)))
    cp.box_json.write_text(json.dumps(box, indent=2))
    return box
