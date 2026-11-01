# Shared context passed between components (plan.md section 4: no cross-imports,
# everything flows through AdapterContext).
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AdapterContext:
    exp_id: str
    components: list[str]
    rank: int
    gralora_k: Optional[int] = None          # set when G present
    # O -> everyone: per-layer rank (LoraConfig rank_pattern / group budget for G)
    layer_ranks: Optional[dict[str, int]] = None
    rank_budget_deviation: int = 0           # param-count drift after k-rounding (X2), recorded in JSON
    # Oi / PiSSA marker: which init was applied (mutually exclusive, X1)
    init_mode: str = "default"               # default | gora | pissa
    # shared runtime state (pipeline fills; hooks read)
    probe_loader: object = None              # N-batch loader for GoRA gradient pre-pass (O/Oi)
    gora_grads: dict = field(default_factory=dict)  # layer_name -> G (CPU fp32), cached O<->Oi
    step: int = 0                            # current global step (BA-LoRA schedules)
    total_steps: int = 0
    # A: pairs of (lora_A_param, lora_B_param) per adapted module, filled after get_peft_model
    ab_pairs: list = field(default_factory=list)
    # B: module name -> forward hook handle for activations
    hooks: dict = field(default_factory=dict)
