"""Stage4/5 combo integration tests (plan.md test layer 2: integration with pinned PEFT).

Replays the exact hook sequence pipeline.build uses for TRIAD (G+Oi+A) and TETRAD-R
(G+Oi+A+Br) on real PEFT GraloraLinear modules, then runs one training step. Guards the
bug class that killed Kaggle session B (trap 13: hook defined but never registered;
trap 14: probe graph cut by checkpointing) before it reaches the GPU. Session-B lesson:
per-method unit tests pass while the combo path is broken — combos need their own test.

TinyV differs from test_gora.Tiny only in the input map: sin(id*c_j) gives full-rank
activations (test_gora.Tiny replicates ONE scalar across features -> rank-1 gram ->
AltLoRA _precond Cholesky fails for shape reasons that do not exist on the real model,
where in_blk >= 224 vs r=32).

Run: python -m pytest tests/unit/test_combos.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # reuse test_gora fixtures

import torch
from torch import nn
from peft import GraloraConfig, get_peft_model

from lorastack.adapters import gora_alloc, gora_init
from lorastack.adapters.base import AdapterContext
from lorastack.config import ExpConfig
from lorastack.losses import balora_reg
from lorastack.registry import get_hooks

from test_gora import DIMS, TARGETS, _loader

_COEFS = torch.linspace(0.2, 1.3, DIMS["q_proj"][0])


class TinyV(nn.Module):
    """test_gora.Tiny with full-rank feature map; forward -> .loss and .logits."""

    def __init__(self):
        super().__init__()
        for name, (din, dout) in DIMS.items():
            setattr(self, name, nn.Linear(din, dout, bias=False))

    def forward(self, input_ids=None, labels=None, **kw):
        h = torch.sin(input_ids.float().unsqueeze(-1) * _COEFS)
        for mod in self.children():
            h = mod(h)
        return SimpleNamespace(loss=((h - labels.float().unsqueeze(-1)) ** 2).mean(), logits=h)


def _combo(components):
    """Build G adapter -> Oi init (own probe) -> A optimizer, pipeline build order."""
    torch.manual_seed(0)
    gora_alloc._G_CACHE.clear()
    peft_model = get_peft_model(
        TinyV(), GraloraConfig(r=8, gralora_k=2, alpha=16, target_modules=list(TARGETS)))
    cfg = ExpConfig(exp_id="combo", components=list(components), rank=8, alpha=16,
                    gora_n_batches=2, lr=1e-3)
    ctx = AdapterContext(exp_id="combo", components=list(components), rank=8)
    ctx.probe_loader = _loader()  # no O in the combo -> Oi runs its own probe (t5b pattern)
    gora_init.init_adapter(cfg, peft_model, ctx)  # raises KeyError if Oi is unregistered (B13)
    assert ctx.init_mode == "gora"
    optimizer = get_hooks("A").make_optimizer(cfg, peft_model, ctx)
    return peft_model, cfg, ctx, optimizer


def _trainables(m):
    return [p for p in m.parameters() if p.requires_grad]


def test_triad_g_oi_a_build_and_step():
    m, cfg, ctx, opt = _combo(["G", "Oi", "A"])
    batch = _loader()[0]
    before = [p.detach().clone() for p in _trainables(m)]
    out = m(input_ids=batch["input_ids"], labels=batch["labels"])
    assert torch.isfinite(out.loss)
    out.loss.backward()
    opt.step()
    moved = any(not torch.equal(p0, p1.detach()) for p0, p1 in zip(before, _trainables(m)))
    assert moved, "one optimizer step must move adapter params"


def test_tetrad_r_g_oi_a_br_reg_terms_flow():
    m, cfg, ctx, _ = _combo(["G", "Oi", "A", "Br"])
    batch = _loader()[0]
    out = m(input_ids=batch["input_ids"], labels=batch["labels"])
    terms = balora_reg.loss_terms(cfg, m, ctx, batch, None, logits_ft=out.logits)
    assert terms, "Br must contribute reg terms on the GraLoRA combo"
    total = out.loss + sum(terms.values())
    assert torch.isfinite(total)
    total.backward()
    grads = [p.grad for p in _trainables(m) if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
