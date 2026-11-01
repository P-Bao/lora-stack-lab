# Component registry: each method registers hooks by its code (L,G,O,Oi,A,B,Bp,Br).
# pipeline.py consumes hooks in the fixed order of plan.md section 4:
#   rank allocation (O) -> adapter build (L|G) -> init (Oi|PiSSA) -> optimizer (A) -> loss (B).
# reimpl modules self-register on first import; pipeline imports them lazily by code.
from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

# code -> SimpleNamespace with any subset of:
#   allocate_ranks(cfg, model, tok, loader) -> dict[layer_name, int]   (O)
#   patch_config(cfg, peft_config)          -> None                    (O: rank_pattern / group budget)
#   init_adapter(cfg, model, ctx)           -> None                    (Oi, Bp/pissa handled by peft)
#   make_optimizer(cfg, model, ctx)         -> torch.optim.Optimizer   (A)
#   loss_terms(cfg, model, ctx, batch)      -> dict[str, torch.Tensor] (B/Bp/Br, summed by pipeline)
COMPONENTS: dict[str, SimpleNamespace] = {}


def register(code: str, **hooks: Callable) -> None:
    if code in COMPONENTS:
        raise ValueError(f"component {code} already registered")
    COMPONENTS[code] = SimpleNamespace(**hooks)


def get_hooks(code: str) -> SimpleNamespace:
    """Lazily import the reimpl module backing `code`, then return its hooks."""
    if code in COMPONENTS:
        return COMPONENTS[code]
    module_by_code = {
        "O": "lorastack.adapters.gora_alloc",
        "Oi": "lorastack.adapters.gora_init",
        "A": "lorastack.optim.altlora",
        "B": "lorastack.losses.balora_reg",
        "Bp": "lorastack.losses.balora_reg",
        "Br": "lorastack.losses.balora_reg",
    }
    if code not in module_by_code:
        raise KeyError(f"unknown component code: {code}")
    __import__(module_by_code[code])  # self-registers
    return COMPONENTS[code]
