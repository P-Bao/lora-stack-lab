"""Plain-assert fallback runner for tests/unit/test_balora.py (no pytest needed).

    python tests/unit/run_balora_checks.py
Exit 0 = all pass; 1 = at least one failure.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_balora as t  # noqa: E402


def main() -> int:
    tests = [(n, getattr(t, n)) for n in sorted(dir(t))
             if n.startswith("test_") and callable(getattr(t, n))]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
