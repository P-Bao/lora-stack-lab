"""Plain-python fallback runner for tests/unit/test_altlora.py (no pytest needed).

Run: python tests/unit/run_altlora_checks.py  — exits 1 if any check fails.
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/ -> import test_altlora
import test_altlora  # noqa: E402


def main() -> int:
    names = sorted(n for n in dir(test_altlora) if n.startswith("test_") and callable(getattr(test_altlora, n)))
    failed = []
    for name in names:
        try:
            getattr(test_altlora, name)()
            print(f"PASS {name}")
        except Exception:
            failed.append(name)
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(names) - len(failed)}/{len(names)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
