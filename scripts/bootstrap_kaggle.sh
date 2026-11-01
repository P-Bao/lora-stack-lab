#!/usr/bin/env bash
# Kaggle bootstrap: install pinned stack (NOT torch — image's torch must stay), verify env, report.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install --quiet -r requirements-kaggle.txt
python tests/test_env.py            # exits non-zero on version mismatch — queue never starts on wrong env
python - <<'PY'
import sys; sys.path.insert(0, "src")
from lorastack.utils.env_report import env_report
print("[env]", env_report())
PY
echo "[bootstrap] OK"
