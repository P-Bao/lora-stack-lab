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
    gora_r_max: Optional[int] = None  # default None = spec/gora.md table
    gora_gamma: Optional[float] = None  # None = spec/gora.md lookup by r_ref
    gora_use_rslora: bool = False     # USER 2026-11-01: alpha/r everywhere; flag for ablation only
    lr_ratio_ba: float = 1.0          # lr multiplier for B params (GoRA App. B.3 uses 16)
    # AltLoRA
    altlora_momentum: float = 0.9
    altlora_order: str = "B_first"    # paper Table 5: B-first 76.31 vs A-first 74.49 (B=0 => A-step no-op)
    altlora_variant: str = "altlora"  # | "altlora_plus" (Algo 2, optional)
    altlora_ridge: float = 1e-6
    # BA-LoRA regularizers — defaults from PAPER (user 2026-11-01); NLG set
    balora_variant: str = "nlg"       # nlg (KL/entropy/randSVD k=10) | nlu (MSE/cov/exactSVD k=5)
    balora_lambda1: float = 1e-5      # consistency (distillation)
    balora_lambda2: float = 1e-1      # diversity (entropy)
    balora_lambda3: float = 1e-1      # SVD-based
    balora_distill_temp: float = 2.0
    balora_top_k_entropy: int = 0     # 0 = full vocab (paper); >0 = top-k proxy
    balora_svd_k: int = 10            # nlu variant uses 5
    balora_schedule: str = "spec42"   # spec/balora.md section 4.2 (paper has no schedule)
    balora_warmup_ratio: float = 0.2
    balora_ramp_up_ratio: float = 0.05

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
    def from_dict(cls, raw: dict) -> "ExpConfig":
        raw = dict(raw or {})
        data_raw = raw.pop("data", None)
        flist = fields(cls)
        known = {f.name for f in flist}
        # YAML 1.1 parses "2e-4" as str (float needs a dot) — coerce str -> int/float
        # by the dataclass annotation so user YAMLs never feed strings into math.
        coerced = {}
        types = {f.name: f.type for f in flist}
        for k, v in raw.items():
            if k in known and isinstance(v, str) and types[k] in ("float", "int",
                                                                  "Optional[float]", "Optional[int]"):
                try:
                    v = float(v) if "float" in types[k] else int(v)
                except ValueError:
                    pass  # keep the string; a clearer error will surface at use site
            coerced[k] = v
        cfg = cls(**{k: v for k, v in coerced.items() if k in known})
        if data_raw:
            cfg.data = DataConfig(**{k: v for k, v in data_raw.items() if k in {f.name for f in fields(DataConfig)}})
        return cfg

    @classmethod
    def from_yaml(cls, path: str, base: Optional[dict] = None) -> "ExpConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        merged = {**(base or {}), **raw}
        return cls.from_dict(merged)

    def dump(self, path: Optional[str] = None) -> dict:
        d: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            d[f.name] = v if f.name != "data" else {ff.name: getattr(v, ff.name) for ff in fields(DataConfig)}
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(d, f, sort_keys=False)
        return d
