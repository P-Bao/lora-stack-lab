# ExpConfig: single source of truth for one experiment run.
# Loaded from configs/experiments/<ID>.yaml by scripts/run_exp.py.
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional

import yaml


@dataclass
class DataConfig:
    name: str = "metamathqa"          # metamathqa | magicoder | glue:<task>
    subset_size: int = 20000
    seq_len: int = 512
    eval_name: str = "gsm8k"          # gsm8k | humaneval | glue:<task>
    eval_size: int = 200


@dataclass
class ExpConfig:
    exp_id: str = "E0-L-qwen05b-lr2e-4-r32-s1"
    # component codes, e.g. ["L"], ["G","O","Oi","A"], ["G","O","A","Br"]
    components: list[str] = field(default_factory=lambda: ["L"])
    impl: dict[str, str] = field(default_factory=dict)  # code -> "peft"|"reimpl" (filled at build)

    model_name: str = "Qwen/Qwen2.5-0.5B"
    # adapter
    rank: int = 32
    gralora_k: int = 2
    alpha: int = 64
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    # GoRA
    gora_n_batches: int = 64          # N for gradient accumulation
    gora_r_min: Optional[int] = None  # default rank//2
    # AltLoRA
    altlora_momentum: float = 0.9
    # BA-LoRA regularizers
    balora_lambda1: float = 0.025     # consistency (distillation)
    balora_lambda2: float = 0.005     # diversity (top-k entropy)
    balora_lambda3: float = 0.005     # SVD-based
    balora_distill_temp: float = 2.0
    balora_top_k_entropy: int = 20
    balora_svd_k: int = 10

    # training
    lr: float = 2e-4
    seed: int = 1
    max_steps: int = -1               # -1 = derive from epochs
    epochs: float = 1.0
    batch_size: int = 8
    grad_accum: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # runtime (T4 fp16)
    fp16: bool = True
    gradient_checkpointing: bool = False
    timeguard_minutes: int = 630      # ~10h30
    ckpt_every_steps: int = 500

    data: DataConfig = field(default_factory=DataConfig)
    out_dir: str = "results"
    ckpt_dir: str = "ckpt"

    @classmethod
    def from_yaml(cls, path: str) -> "ExpConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        data_raw = raw.pop("data", None)
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in known}
        cfg = cls(**filtered)
        if data_raw:
            cfg.data = DataConfig(**{k: v for k, v in data_raw.items() if k in {f.name for f in fields(DataConfig)}})
        return cfg

    def dump(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        d: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            d[f.name] = v if f.name != "data" else {ff.name: getattr(v, ff.name) for ff in fields(DataConfig)}
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(d, f, sort_keys=False)
