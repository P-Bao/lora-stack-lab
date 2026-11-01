#!/usr/bin/env python3
"""Zip results dir for manual download from Kaggle. Also invoked by hand after TimeGuard stops."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/kaggle/working/results")
    ap.add_argument("--zip", default="/kaggle/working/results_pack.zip")
    args = ap.parse_args()
    src = Path(args.src)
    if not src.exists():
        print(f"[collect] {src} missing, nothing to do")
        return 1
    n = 0
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(src.parent))
                n += 1
    print(f"[collect] {n} files -> {args.zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
