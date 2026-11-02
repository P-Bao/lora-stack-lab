#!/usr/bin/env bash
# Kaggle bootstrap: install pinned stack (NOT torch — image's torch must stay), verify env, report.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # T4: fight fragmentation (2.3 GiB logits alloc)

python -m pip install --quiet -r requirements-kaggle.txt
# Kaggle image ships torchao 0.10.0; peft >=0.19 RAISES on old torchao even when unused
# (is_torchao_available raises instead of returning False). pip uninstall refuses
# outside-environment dists (/usr/lib/python3/dist-packages) -> sweep ALL sys.path dirs.
python - <<'PY'
import importlib, importlib.util as iu, pathlib, shutil, sys
removed = []
for sp in list(sys.path):
    if not sp:
        continue
    spp = pathlib.Path(sp)
    if not spp.is_dir():
        continue
    for d in spp.glob("torchao*"):  # package dir, dist-info, egg-info, egg-link, .pth
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
        else:
            d.unlink(missing_ok=True)
        removed.append(str(d))
importlib.invalidate_caches()
spec = iu.find_spec("torchao")
if spec is not None:
    sys.exit(f"[bootstrap] FATAL: torchao still importable at {spec.origin} after removing {removed}")
print(f"[bootstrap] torchao: removed {len(removed)} entries, now absent - OK")
PY
python tests/test_env.py            # exits non-zero on version mismatch — queue never starts on wrong env
python - <<'PY'
import sys; sys.path.insert(0, "src")
from lorastack.utils.env_report import env_report
print("[env]", env_report())
PY
echo "[bootstrap] OK"
