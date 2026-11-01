#!/usr/bin/env bash
# Kaggle bootstrap: install pinned stack (NOT torch — image's torch must stay), verify env, report.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install --quiet -r requirements-kaggle.txt
# Kaggle image ships torchao 0.10.0; peft >=0.19 RAISES on old torchao even when unused
# (is_torchao_available raises instead of returning False). We never use torchao -> remove it.
python -m pip uninstall -y torchao 2>/dev/null || true
python tests/test_env.py            # exits non-zero on version mismatch — queue never starts on wrong env
python - <<'PY'
import sys; sys.path.insert(0, "src")
from lorastack.utils.env_report import env_report
print("[env]", env_report())
PY
echo "[bootstrap] OK"
