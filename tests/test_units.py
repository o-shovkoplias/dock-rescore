"""Fast unit tests that only need RDKit + numpy (run: python -m pytest -q tests)."""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from dockrescore.features import RDKIT_2D, ligand_descriptors
from dockrescore.prepare import docking_box
from dockrescore.rmsd import symm_rmsd


def _mol(smiles: str, seed: int = 1) -> Chem.Mol:
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    return m


def test_docking_box_padding() -> None:
    m = _mol("c1ccccc1O")
    box = docking_box(m, padding=8.0, min_size=20.0)
    assert len(box["center"]) == 3 and all(s >= 20.0 for s in box["size"])


def test_symmetric_rmsd_is_zero_for_permuted_copy() -> None:
    m = _mol("c1ccccc1C(=O)O")
    # renumber atoms: symmetry-aware RMSD must still be ~0 without alignment
    perm = list(range(m.GetNumAtoms()))[::-1]
    m2 = Chem.RenumberAtoms(m, perm)
    assert symm_rmsd(m2, m) < 1e-6


def test_rmsd_detects_translation() -> None:
    m = _mol("CCO")
    m2 = Chem.Mol(m)
    conf = m2.GetConformer()
    for i in range(m2.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + 3.0, p.y, p.z))
    assert abs(symm_rmsd(m2, m) - 3.0) < 1e-6


def test_descriptor_list_is_fixed_and_finite() -> None:
    d = ligand_descriptors(_mol("CC(=O)Nc1ccc(O)cc1"))
    assert len(RDKIT_2D) == 40 and len(d) == 40
    assert np.isfinite(list(d.values())).all()
