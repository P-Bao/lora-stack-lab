#!/usr/bin/env python3
"""Run ONE experiment by config YAML. Writes results/<ID>.json on success,
results/<ID>.FAILED.json (with traceback) on error — the queue never dies with one run."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import yaml

from lorastack.config import ExpConfig  # noqa: E402
from lorastack.pipeline import run_experiment  # noqa: E402
from lorastack.utils.logging import write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to experiment YAML")
    ap.add_argument("--out", default=None, help="override results dir")
    ap.add_argument("--ckpt", default=None, help="override ckpt dir")
    args = ap.parse_args()

    base_path = REPO / "configs" / "base.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}
    cfg = ExpConfig.from_yaml(args.config, base=base)
    if args.out:
        cfg.out_dir = args.out
    if args.ckpt:
        cfg.ckpt_dir = args.ckpt
    out_dir = Path(cfg.out_dir)
    done_marker = out_dir / f"{cfg.exp_id}.json"
    if done_marker.exists():
        print(f"[run_exp] {cfg.exp_id} already done, skip")
        return 0

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        result = run_experiment(cfg)
        write_json(str(done_marker), result)
        print(f"[run_exp] {cfg.exp_id} -> {result.get('status')} metrics={result.get('metrics')}")
        return 0
    except Exception:
        write_json(str(out_dir / f"{cfg.exp_id}.FAILED.json"),
                   {"id": cfg.exp_id, "status": "error", "traceback": traceback.format_exc(),
                    "config": cfg.dump()})
        raise


if __name__ == "__main__":
    sys.exit(main())
