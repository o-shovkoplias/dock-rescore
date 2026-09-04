#!/usr/bin/env python
"""Build data/manifest.csv from the unpacked PoseBusters Benchmark set.

One row per complex: identifiers, ligand size, SMILES (if RDKit is importable),
protein size, and membership in the 308-complex journal subset. Runs from the repo root:

    python scripts/make_manifest.py [--config config/config.yaml]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

try:  # RDKit is optional here so the manifest can be rebuilt in a minimal env
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:  # pragma: no cover
    Chem = None


def sdf_heavy_atoms(path: Path) -> int:
    """Count non-hydrogen atoms in the first molecule of an SDF file (V2000 or V3000)."""
    lines = path.read_text().splitlines()
    if len(lines) > 3 and "V3000" in lines[3]:
        return sum(
            1
            for ln in lines
            if ln.startswith("M  V30 ") and len(ln.split()) > 4 and ln.split()[3] != "H"
            and ln.split()[2].isdigit()
        )
    n_atoms = int(lines[3][:3])
    return sum(1 for ln in lines[4 : 4 + n_atoms] if ln[31:34].strip() != "H")


def protein_stats(path: Path) -> tuple[int, int, int, int]:
    """Return (#ATOM records, #HETATM records, #chains, #distinct HETATM residue names)."""
    n_atom = n_het = 0
    chains: set[str] = set()
    het: set[str] = set()
    with path.open() as fh:
        for ln in fh:
            if ln.startswith("ATOM"):
                n_atom += 1
                chains.add(ln[21])
            elif ln.startswith("HETATM"):
                n_het += 1
                het.add(ln[17:20].strip())
    return n_atom, n_het, len(chains), len(het)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    raw = Path(cfg["paths"]["raw_dir"])
    ids_308 = set(Path(cfg["paths"]["ids_308"]).read_text().split())
    rows = []
    for d in sorted(p for p in raw.iterdir() if p.is_dir()):
        cid = d.name
        lig = d / f"{cid}_ligand.sdf"
        prot = d / f"{cid}_protein.pdb"
        smiles = ""
        if Chem is not None:
            mol = Chem.MolFromMolFile(str(lig))
            smiles = Chem.MolToSmiles(mol) if mol is not None else "PARSE_FAIL"
        n_atom, n_het, n_chains, n_het_res = protein_stats(prot)
        rows.append(
            {
                "complex_id": cid,
                "pdb_id": cid.split("_")[0],
                "ccd_id": cid.split("_", 1)[1],
                "in_308_subset": int(cid in ids_308),
                "ligand_heavy_atoms": sdf_heavy_atoms(lig),
                "ligand_smiles": smiles,
                "protein_atoms": n_atom,
                "protein_chains": n_chains,
                "hetatm_atoms": n_het,
                "hetatm_residue_types": n_het_res,
            }
        )
    out = Path(cfg["paths"]["manifest"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[make_manifest] wrote {out} with {len(rows)} complexes "
          f"({sum(r['in_308_subset'] for r in rows)} in the 308 subset); "
          f"smiles={'rdkit' if Chem else 'skipped (no rdkit)'}")


if __name__ == "__main__":
    main()
