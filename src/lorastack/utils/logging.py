"""JSONL step logger + thin-thick result writing (plan.md 5.5)."""
from __future__ import annotations

import json
import os
from typing import Optional


class JsonlLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def log(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)  # atomic: queue/collect never sees a half-written JSON


def read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
