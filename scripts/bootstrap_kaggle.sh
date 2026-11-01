#!/usr/bin/env bash
# Kaggle bootstrap: install pinned stack (NOT torch — image's torch must stay), verify env, report.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install --quiet -r requirements-kaggle.txt
# Kaggle image ships torchao 0.10.0; peft >=0.19 RAISES on old torchao even when unused
# (is_torchao_available raises instead of returning False). pip uninstall refuses
# outside-environment dists (/usr/lib/python3/dist-packages) -> remove files directly.
python - <<'PY'
import importlib.util as iu, pathlib, shutil, sys
spec = iu.find_spec("torchao")
if spec is None:
    print("[bootstrap] torchao absent - OK")
else:
    for loc in (spec.submodule_search_locations or []):
        print("[bootstrap] removing", loc); shutil.rmtree(loc, ignore_errors=True)
    if spec.origin:
        pathlib.Path(spec.origin).unlink(missing_ok=True)
    for sp in sys.path:
        if sp and (spp := pathlib.Path(sp)).is_dir():
            for d in spp.glob("torchao-*.dist-info"):
                shutil.rmtree(d, ignore_errors=True)
    if iu.find_spec("torchao") is not None:
        sys.exit("[bootstrap] FATAL: torchao still importable after removal")
    print("[bootstrap] torchao removed")
PY
python tests/test_env.py            # exits non-zero on version mismatch — queue never starts on wrong env
python - <<'PY'
import sys; sys.path.insert(0, "src")
from lorastack.utils.env_report import env_report
print("[env]", env_report())
PY
echo "[bootstrap] OK"
