#!/usr/bin/env python3
"""Local aggregation (no GPU): tables + interaction effects from results JSONs.
Δ_int(X,Y) = S(X+Y) − S(X) − S(Y) + S(L*)   (plan.md section 7, stage 3)"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from lorastack.utils.logging import read_json  # noqa: E402


def score(rec: dict) -> float | None:
    m = rec.get("metrics") or {}
    for k, v in m.items():
        if k.endswith("_acc"):
            return v * 100.0
    return None


def combo_key(rec: dict) -> str:
    cfg = rec.get("config")
    return "+".join(cfg["components"]) if isinstance(cfg, dict) and "components" in cfg else rec["id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    groups: dict[str, list[float]] = defaultdict(list)
    for f in sorted(Path(args.results).rglob("*.json")):
        if f.name.endswith(".FAILED.json") or f.name.endswith(".config.yaml") or f.name.endswith(".log.jsonl"):
            continue
        rec = read_json(str(f))
        if not rec or rec.get("status") != "done":
            continue
        s = score(rec)
        if s is not None:
            groups[combo_key(rec)].append(s)

    print(f"{'combo':30s} {'n':>2s} {'mean':>7s} {'std':>6s}")
    means = {}
    for combo, vals in sorted(groups.items()):
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        means[combo] = mean
        print(f"{combo:30s} {len(vals):2d} {mean:7.2f} {std:6.2f}")

    base = means.get("L")
    if base is None:
        print("\n(no L baseline found — skip Δ columns)")
        return 0
    print("\nΔ vs L* (project threshold: max(1.0 pt, 2·σ_L)):")
    for combo, mean in sorted(means.items()):
        if combo != "L":
            print(f"  {combo:28s} {mean - base:+7.2f}")
    combos = [c for c in means if c != "L"]
    for x in combos:
        for y in combos:
            pair = f"{x}+{y}"
            if x < y and pair in means:  # pair must exist as its own combo
                d = means[pair] - means[x] - means[y] + base
                print(f"  Δ_int({x},{y}) = {d:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
