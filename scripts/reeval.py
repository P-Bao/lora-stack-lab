#!/usr/bin/env python3
"""Re-evaluate saved adapters WITHOUT retraining (stage1 rescue: gsm8k_acc=0.0 bug + missing lr5e-5 json).

Adapter layout (Kaggle Dataset or local): any depth, dirs of the form <run_id>/adapter/adapter_config.json.
Writes <out>/<run_id>.reeval.json (metrics, same shape as run JSONs) and
<out>/<run_id>.reeval.jsonl (per-sample generations for diagnosis).

Usage:
  python scripts/reeval.py --adapters /kaggle/input/stage1-adapters --out /kaggle/working/results
  python scripts/reeval.py --adapters wt-results/stage1/ckpt --limit 4 --batch-size 4   # local smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lorastack.config import ExpConfig  # noqa: E402
from lorastack.eval import evaluate  # noqa: E402
from lorastack.pipeline import _git_commit  # noqa: E402
from lorastack.utils.env_report import env_report  # noqa: E402
from lorastack.utils.logging import write_json  # noqa: E402


def find_adapter_runs(root: str) -> list[tuple[str, Path]]:
    """Returns [(run_id, adapter_dir)]; run_id = name of the dir containing 'adapter'."""
    runs = []
    for cfg_path in sorted(Path(root).rglob("adapter_config.json")):
        run_dir = cfg_path.parent.parent
        runs.append((run_dir.name, cfg_path.parent))
    ids = [rid for rid, _ in runs]
    dupes = sorted({rid for rid in ids if ids.count(rid) > 1})
    if dupes:
        # flattened upload (<root>/adapter/) makes every run id "adapter" -> silent overwrite
        raise SystemExit(f"[reeval] duplicate run ids {dupes} — dataset layout must be <run_id>/adapter/")
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", required=True, help="root dir containing <run_id>/adapter/")
    ap.add_argument("--out", default="results")
    ap.add_argument("--eval-size", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--components", default="L", help="comma-joined codes for aggregate.py combo key")
    ap.add_argument("--shard", default="0/1", help="i/n: handle runs with index %% n == i (2-GPU split)")
    ap.add_argument("--limit", type=int, default=0, help=">0 = eval only the first N prompts (smoke)")
    ap.add_argument("--base", default=None, help="override base model (default from ExpConfig)")
    args = ap.parse_args()

    runs = find_adapter_runs(args.adapters)
    if not runs:
        print(f"[reeval] no adapter dirs under {args.adapters}")
        return 1
    si, sn = (int(x) for x in args.shard.split("/"))
    mine = runs[si::sn]
    print(f"[reeval] shard {si}/{sn}: {len(mine)}/{len(runs)} runs: {[r for r, _ in mine]}", flush=True)
    runs = mine
    if not runs:
        return 0
    print(f"[reeval] {len(runs)} adapters: {[rid for rid, _ in runs]}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    for run_id, adapter_dir in runs:
        done = out_dir / f"{run_id}.reeval.json"
        if done.exists():
            print(f"[reeval] {run_id} already done, skip", flush=True)
            continue
        t0 = time.time()
        cfg = ExpConfig(exp_id=run_id, components=args.components.split(","))
        cfg.data.eval_size = args.eval_size
        cfg.data.eval_batch_size = args.batch_size
        base = args.base or cfg.model_name

        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model = model.merge_and_unload()  # LoRA merge: identical outputs, faster generate
        if args.limit > 0:
            cfg.data.eval_size = args.limit

        samples: list = []
        metrics = evaluate(model, tok, cfg, device, collect=samples)
        # adapter snapshot is step ckpt_every_steps (500), not final step — see decisions.md
        rec = {
            "status": "done",
            "id": run_id,
            "reeval": True,
            "adapter_step": "ckpt_every_steps snapshot",
            "adapter_path": str(adapter_dir),
            "metrics": metrics,
            "git_commit": _git_commit(),
            "env": env_report(),
            "wall_time_s": round(time.time() - t0, 1),
            "config": {"components": cfg.components},  # combo key for aggregate.py
        }
        with open(out_dir / f"{run_id}.reeval.jsonl", "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        # json last: its presence marks the run complete incl. the generation dump
        write_json(str(done), rec)
        n_digit = sum(1 for s in samples if s["pred"] is not None)
        print(f"[reeval] {run_id} acc={metrics['gsm8k_acc']:.4f} "
              f"preds_with_digits={n_digit}/{len(samples)} {rec['wall_time_s']}s", flush=True)
        del model
        torch.cuda.empty_cache() if device == "cuda" else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
