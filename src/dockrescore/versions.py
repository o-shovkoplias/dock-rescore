"""Record the exact versions of the chemistry stack used for a run."""
from __future__ import annotations

import importlib
import platform
import sys
from typing import Any


def collect_versions() -> dict[str, Any]:
    """Return {package: version | 'MISSING: <error>'} for the packages that matter."""
    out: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod, attr in [
        ("rdkit", "__version__"),
        ("vina", "__version__"),
        ("meeko", "__version__"),
        ("openbabel", "__version__"),
        ("prolif", "__version__"),
        ("MDAnalysis", "__version__"),
        ("torch", "__version__"),
        ("sklearn", "__version__"),
        ("numpy", "__version__"),
        ("pandas", "__version__"),
    ]:
        try:
            m = importlib.import_module(mod)
            v = getattr(m, attr, None)
            if mod == "openbabel" and v is None:  # openbabel exposes the version differently
                from openbabel import openbabel as ob

                v = ob.OBReleaseVersion()
            if mod == "vina" and v is None:
                from vina import Vina

                v = Vina(verbosity=0).__str__().split("\n")[0]
            out[mod] = str(v)
        except Exception as e:  # noqa: BLE001
            out[mod] = f"MISSING: {type(e).__name__}: {e}"
    return out
