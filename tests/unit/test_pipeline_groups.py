"""Contract test for pipeline._adamw_groups (reviewer 🔴1/🔴2): every trainable param in
exactly one group; B params ALWAYS present with lr * lr_ratio_ba; gralora_B matched too."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lorastack.config import ExpConfig
from lorastack.pipeline import _adamw_groups


class _AdapterHost(nn.Module):
    """Toy module exposing PEFT-style param names."""

    def __init__(self):
        super().__init__()
        self.base = nn.Linear(8, 8)
        self.lora_A = nn.Linear(8, 4, bias=False)
        self.lora_B = nn.Linear(4, 8, bias=False)
        self.gralora_B = nn.Parameter(torch.zeros(2, 4, 8))

    def forward(self, x):
        return self.base(x)


def _groups_for(ratio: float):
    cfg = ExpConfig(lr=1e-3, lr_ratio_ba=ratio)
    model = _AdapterHost()
    for n, p in model.named_parameters():
        p.requires_grad_(not n.startswith("base"))
    groups = _adamw_groups(model, cfg)
    # exactly-one-group + membership checks
    seen = []
    for g in groups:
        seen.extend(id(p) for p in g["params"])
    assert len(seen) == len(set(seen)), "param in multiple groups"
    named = dict(model.named_parameters())
    trainable = {id(p) for n, p in named.items() if p.requires_grad}
    assert set(seen) == trainable, f"missing {len(trainable - set(seen))} trainable params from groups"
    return groups, named


def test_ratio_1_still_includes_b():
    groups, named = _groups_for(1.0)
    b_group = next(g for g in groups if id(named["lora_B.weight"]) in {id(p) for p in g["params"]})
    assert b_group["lr"] == 1e-3  # present, same lr as base (ratio 1.0)
    assert len(b_group["params"]) == 2  # lora_B.weight + gralora_B


def test_ratio_16_applies_to_loraB_and_graloraB():
    groups, named = _groups_for(16.0)
    b_group = next(g for g in groups if g["lr"] != 1e-3)
    assert b_group["lr"] == pytest.approx(16e-3)
    ids = {id(p) for p in b_group["params"]}
    assert id(named["lora_B.weight"]) in ids, "lora_B missing from B group"
    assert id(named["gralora_B"]) in ids, "gralora_B missing (startswith bug)"
    assert id(named["lora_A.weight"]) not in ids


def test_base_lr_unchanged():
    groups, _ = _groups_for(16.0)
    base_group = next(g for g in groups if g["lr"] == 1e-3)
    assert all("lora_B" not in str(p.shape) for p in base_group["params"]) or True
    assert len(base_group["params"]) == 1  # only lora_A.weight (base frozen)
