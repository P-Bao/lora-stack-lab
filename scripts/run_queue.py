#!/usr/bin/env python3
"""Queue runner: 2 experiments in parallel (one per T4), skip finished IDs,
scan previous outputs (--skip-from) so re-attached Kaggle inputs resume cleanly."""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def find_configs(stage: str | None, ids: list[str]) -> list[Path]:
    exp_dir = REPO / "configs" / "experiments"
    all_yml = sorted(exp_dir.glob("*.yaml"))
    if ids:
        want = set(ids)
        # config stem is a PREFIX of the full experiment ID (E0-SMOKE-L -> E0-SMOKE-L-qwen05b-...)
        return [p for p in all_yml if p.stem in want or any(w.startswith(p.stem) for w in want)]
    if stage:
        return [p for p in all_yml if p.stem.startswith(f"{stage}-")]
    return all_yml


def done_ids(search_dirs: list[str]) -> set[str]:
    done = set()
    for root in search_dirs:
        if not root or not Path(root).exists():
            continue
        for base, _, files in os.walk(root):
            for f in files:
                if f.endswith(".json") and not f.endswith((".FAILED.json", ".config.yaml", ".log.jsonl")):
                    done.add(f[: -len(".json")])
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None)
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--out", default="/kaggle/working/results")
    ap.add_argument("--ckpt", default="/kaggle/working/ckpt")
    ap.add_argument("--skip-from", nargs="*", default=[], help="dirs (old outputs mounted as input) to scan for done IDs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)  # collect_results needs the dir even with 0 runs
    todo = [p for p in find_configs(args.stage, args.ids) if p.stem not in done_ids([args.out, *args.skip_from])]
    if not todo:
        print(f"[run_queue] NOTHING matched (stage={args.stage!r}, ids={args.ids}) "
              f"— available: {[p.stem for p in find_configs(args.stage, [])] or find_configs(None, [])}")
    print(f"[run_queue] {len(todo)} to run: {[p.stem for p in todo]}")
    gpu_cycle = itertools.cycle(["0", "1"])
    running: dict[subprocess.Popen, str] = {}
    failures = 0

    def reap(block: bool) -> None:
        nonlocal failures
        while running:
            for p in list(running):
                code = p.poll()
                if code is not None:
                    failures += int(code != 0)
                    # "-u" shifts arg indices — locate --config positionally
                    idx = p.args.index("--config")
                    print(f"[run_queue] {Path(p.args[idx + 1]).stem} exit={code}")
                    running.pop(p)
            if not running:
                break
            if not block:
                return
            time.sleep(10)

    for cfg_path in todo:
        while len(running) >= 2:  # one per T4
            reap(block=False)
            time.sleep(5)
        gpu = next(gpu_cycle)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        p = subprocess.Popen(
            [sys.executable, "-u", str(REPO / "scripts" / "run_exp.py"), "--config", str(cfg_path),
             "--out", args.out, "--ckpt", args.ckpt],
            env=env,
        )
        running[p] = gpu
        print(f"[run_queue] start {cfg_path.stem} on GPU {gpu}")
    reap(block=True)
    print(f"[run_queue] finished, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
