#!/usr/bin/env bash
# Download the PoseBusters Benchmark set (Buttenschoen, Morris, Deane 2024) from Zenodo,
# verify the MD5 checksum published by the Zenodo API, and unpack it into data/raw/.
# Usage: bash scripts/fetch_data.sh        (run from the repository root)
set -euo pipefail
cd "$(dirname "$0")/.."

RECORD=8278563                       # zenodo.org/records/8278563  (arXiv:2308.05777 data)
FILE=posebusters_paper_data.zip
RAW=data/raw
mkdir -p "$RAW"

PY=${PYTHON:-python}

echo "[fetch_data] querying Zenodo record $RECORD"
"$PY" - "$RECORD" "$FILE" "$RAW" <<'PYEOF'
import hashlib, json, pathlib, sys, urllib.request
record, fname, raw = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
rec = json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=120))
f = next(x for x in rec["files"] if x["key"] == fname)
url, md5 = f["links"]["self"], f["checksum"].split(":")[1]
dst = raw / fname
if not dst.exists() or hashlib.md5(dst.read_bytes()).hexdigest() != md5:
    print(f"[fetch_data] downloading {url} ({f['size']/1e6:.1f} MB)")
    urllib.request.urlretrieve(url, dst)
got = hashlib.md5(dst.read_bytes()).hexdigest()
if got != md5:
    sys.exit(f"[fetch_data] MD5 mismatch: {got} != {md5}")
print(f"[fetch_data] md5 ok: {md5}")
(raw / "CHECKSUMS.md5").write_text(f"{md5}  {fname}\n")
PYEOF

echo "[fetch_data] unpacking"
unzip -q -o "$RAW/$FILE" -d "$RAW"

# 308-complex subset used in the journal version of the paper (crystal-contact-free)
IDS308="$RAW/posebusters_pdb_ccd_ids_308.txt"
if [ ! -s "$IDS308" ]; then
  echo "[fetch_data] fetching 308-id subset list"
  "$PY" -c "import urllib.request,sys; open(sys.argv[1],'wb').write(urllib.request.urlopen('https://github.com/maabuu/posebusters/files/14516485/posebusters_pdb_ccd_ids.txt',timeout=120).read())" "$IDS308"
fi
echo "[fetch_data] complexes: $(ls -d $RAW/posebusters_benchmark_set/*/ | wc -l) (428 expected), 308-subset ids: $(wc -w < "$IDS308")"
