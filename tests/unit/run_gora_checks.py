"""Fallback runner without pytest: python tests/unit/run_gora_checks.py"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_gora  # noqa: E402  (bootstraps src/ onto sys.path)


def main() -> int:
    names = sorted(n for n in dir(test_gora) if n.startswith("test_") and callable(getattr(test_gora, n)))
    failed = 0
    for name in names:
        try:
            getattr(test_gora, name)()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(names) - failed}/{len(names)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
