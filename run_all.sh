#!/usr/bin/env bash
# Single entry point. Runs from the repository root inside the "dock" conda env.
#   bash run_all.sh                 # pilot (data/pilot_ids.txt)
#   IDS=data/ids_308.txt bash run_all.sh   # full 308-complex run (phase 2; hours of CPU docking)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:$PWD/src"
PY=${PYTHON:-python}
IDS=${IDS:-data/pilot_ids.txt}

echo "== dock-rescore run_all.sh =="
echo "python: $($PY -c 'import sys; print(sys.executable)')"
if ! $PY -c "import vina, meeko, rdkit, prolif, torch" 2>/tmp/dockrescore_import_err.txt; then
  echo "!! The chemistry stack is not importable in this python:"; cat /tmp/dockrescore_import_err.txt
  echo "!! Activate the 'dock' env (mamba env create -f environment.yml) and re-run. Stopping."
  exit 2
fi

echo "[1/8] fetch data (skipped if md5-verified archive is already present)"
PYTHON=$PY bash scripts/fetch_data.sh
echo "[2/8] manifest"
$PY scripts/make_manifest.py
echo "[3/8] record versions + select pilot complexes"
$PY -m dockrescore versions
[ -s "$IDS" ] || $PY -m dockrescore select-pilot
echo "[4/8] prepare receptors/ligands (ids: $IDS)"
$PY -m dockrescore prepare --ids "$IDS"
echo "[5/8] dock with AutoDock Vina (CPU; exhaustiveness from config/config.yaml)"
$PY -m dockrescore dock --ids "$IDS"
echo "[6/8] RMSD labels + features"
$PY -m dockrescore rmsd --ids "$IDS"
$PY -m dockrescore features --ids "$IDS"
echo "[7/8] train/evaluate MLP (GroupKFold by complex, CPU)"
$PY -m dockrescore train
echo "[8/8] figures + report"
$PY -m dockrescore report --ids "$IDS"
$PY scripts/update_readme.py
echo "== done: results/metrics.json, results/report.md, figures/top1_success.png =="
