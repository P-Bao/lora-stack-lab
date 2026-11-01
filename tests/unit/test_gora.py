"""GoRA unit tests (CPU, tiny tensors; local peft 0.20.0 — never assert pinned versions).

T1 projection init (A·B ≈ -P_A·G) | T2 budget/round/clamp/rescale | T3 rank_pattern + PEFT
consumes it + GraloraConfig deviation | T4 GraLoRA k=1 == full-matrix init (X3) |
T5 shared G cache O→Oi without recompute (+ Oi standalone probe).
Run: python -m pytest tests/unit/test_gora.py -q   (fallback: python tests/unit/run_gora_checks.py)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch import nn
from peft import GraloraConfig, LoraConfig, get_peft_model

from lorastack.adapters import gora_alloc, gora_init
from lorastack.adapters.base import AdapterContext
from lorastack.config import ExpConfig

DIMS = {"q_proj": (16, 8), "k_proj": (8, 8), "v_proj": (8, 12), "down_proj": (12, 4)}
TARGETS = ["q_proj", "v_proj"]


class Tiny(nn.Module):
    """4-Linear chain; forward(input_ids, labels) -> object with .loss (CPU test stand-in)."""

    def __init__(self):
        super().__init__()
        for name, (din, dout) in DIMS.items():
            setattr(self, name, nn.Linear(din, dout, bias=False))

    def forward(self, input_ids=None, labels=None, **kw):
        h = input_ids.float().unsqueeze(-1).expand(-1, -1, DIMS["q_proj"][0])  # (B, L, in0)
        for mod in self.children():
            h = mod(h)
        return SimpleNamespace(loss=((h - labels.float().unsqueeze(-1)) ** 2).mean())


class One(nn.Module):
    """Single target layer for init tests."""

    def __init__(self, din=12, dout=10):
        super().__init__()
        self.q_proj = nn.Linear(din, dout, bias=False)


def _loader(n_batches=2, batch=2, length=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        {"input_ids": torch.randint(0, 100, (batch, length), generator=g),
         "labels": torch.randn(batch, length, generator=g)}
        for _ in range(n_batches)
    ]


def _cfg(**kw):
    d = dict(components=["O"], rank=8, alpha=16, gora_n_batches=2, target_modules=list(TARGETS))
    d.update(kw)
    return ExpConfig(**d)


# ---------------------------------------------------------------- T1: projection init

def _setup_lora_t1(fp16_b=False):
    torch.manual_seed(0)
    m, n, r = 10, 12, 4
    peft_model = get_peft_model(One(), LoraConfig(r=r, lora_alpha=8, target_modules=["q_proj"]))
    layer = peft_model.base_model.model.q_proj
    M = torch.randn(r, n, generator=torch.Generator().manual_seed(1))
    layer.lora_A["default"].weight.data.copy_(M)
    if fp16_b:
        layer.lora_B["default"].weight.data = layer.lora_B["default"].weight.data.half()
    G = (torch.randn(m, 4, generator=torch.Generator().manual_seed(2))
         @ torch.randn(4, n, generator=torch.Generator().manual_seed(3)))
    gora_alloc._G_CACHE.clear()
    gora_alloc._G_CACHE["q_proj"] = G.float()
    # gamma chosen so xi = gamma*sqrt(n)/(s*sqrt(r)) == 1  (s = alpha/r = 2)
    cfg = _cfg(components=["Oi"], rank=r, alpha=8, gora_gamma=(8 / r) * math.sqrt(r) / math.sqrt(n))
    ctx = AdapterContext(exp_id="t1", components=["Oi"], rank=r)
    gora_init.init_adapter(cfg, peft_model, ctx)
    return layer, M, G, ctx


def test_t1_projection_init():
    layer, M, G, ctx = _setup_lora_t1()
    B = layer.lora_B["default"].weight.double()  # (m, r)
    A = layer.lora_A["default"].weight.double()  # (r, n)
    # P_row computed independently in fp64, no ridge (well-conditioned -> ridge negligible)
    Md = M.double()
    P = Md.T @ torch.linalg.inv(Md @ Md.T) @ Md
    resid = (B @ A + G.double() @ P).norm() / (G.double() @ P).norm()
    assert resid < 1e-4, f"projection residual {resid:.2e}"
    assert torch.allclose(A, M.double()), "lora_A must stay untouched (A0 kept)"
    assert ctx.init_mode == "gora"


def test_t1b_dtype_preserved_fp16():
    layer, _, G, _ = _setup_lora_t1(fp16_b=True)
    B = layer.lora_B["default"].weight
    assert B.dtype == torch.float16
    # fp32 reference of the same assignment
    torch.manual_seed(0)
    layer32, M, _, _ = _setup_lora_t1(fp16_b=False)
    ref = layer32.lora_B["default"].weight.double()
    assert (B.double() - ref).norm() / ref.norm() < 2e-3  # fp16 quantization only


# ---------------------------------------------------------------- T2: budget / clamp / rescale

def test_t2_budget_round_clamp_rescale():
    shapes = {"a": (20, 20), "b": (20, 20), "c": (20, 20)}  # m+n = 40 each
    cfg = _cfg(rank=8)
    # equal importance + equal shapes -> every r = r_ref (= plain-LoRA allocation)
    ranks = gora_alloc._allocate_ranks_from_importance({"a": 1.0, "b": 1.0, "c": 1.0}, shapes, cfg)
    assert ranks == {"a": 8, "b": 8, "c": 8}

    # skewed importance hits r_min on two layers -> 16.7% drift -> one rescale round
    ranks = gora_alloc._allocate_ranks_from_importance({"a": 10.0, "b": 1.0, "c": 1.0}, shapes, cfg)
    r_min, r_max = 4, 32  # r_ref/2, 4*r_ref
    assert all(r_min <= r <= r_max for r in ranks.values()), ranks
    B_tot = sum(m + n for m, n in shapes.values()) * 8
    P = sum((shapes[k][0] + shapes[k][1]) * r for k, r in ranks.items())
    assert abs(P - B_tot) / B_tot <= 0.10, f"budget drift {abs(P - B_tot) / B_tot:.1%}"
    assert ranks["a"] > ranks["b"]  # monotone in importance


def test_t2b_allocate_end_to_end():
    torch.manual_seed(0)
    model = Tiny()
    cfg = _cfg(rank=8, gora_n_batches=2)
    ranks = gora_alloc.allocate_ranks(cfg, model, None, _loader())
    assert set(ranks) == set(TARGETS)
    assert all(4 <= r <= 32 for r in ranks.values())
    cache = gora_alloc.get_cached_grads()
    assert set(cache) == set(TARGETS)
    params = dict(model.named_parameters())
    for name, g in cache.items():
        assert g.dtype == torch.float32 and g.device.type == "cpu"
        assert tuple(g.shape) == tuple(params[name + ".weight"].shape)
    # probe left the model clean: original requires_grad restored, no stray grads (spec 7.5)
    assert all(p.requires_grad for n, p in params.items() if n.startswith(("q_proj", "v_proj")))
    assert all(p.grad is None for p in params.values())


# ---------------------------------------------------------------- T3: patch_config

def test_t3_patch_config_lora_and_peft_consumes():
    torch.manual_seed(0)
    model = Tiny()
    cfg = _cfg(rank=8, alpha=16)
    ctx = AdapterContext(exp_id="t3", components=["O", "L"], rank=8)
    ctx.layer_ranks = gora_alloc.allocate_ranks(cfg, model, None, _loader())
    peft_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=list(TARGETS))
    gora_alloc.patch_config(cfg, peft_cfg, ctx)
    assert len(peft_cfg.rank_pattern) == 2
    for name, r in ctx.layer_ranks.items():
        assert peft_cfg.rank_pattern[name] == r
        # alpha_pattern keeps per-layer effective scaling s = alpha/r_ref = 2.0
        assert abs(peft_cfg.alpha_pattern[name] - 2.0 * r) < 1e-9
    # spec 9.11 smoke: PEFT must actually apply the per-layer rank through the pattern
    wrapped = get_peft_model(Tiny(), peft_cfg)
    assert wrapped.base_model.model.q_proj.lora_A["default"].weight.shape[0] == ctx.layer_ranks["q_proj"]
    assert wrapped.base_model.model.v_proj.lora_A["default"].weight.shape[0] == ctx.layer_ranks["v_proj"]


def test_t3b_patch_config_gralora_deviation():
    torch.manual_seed(0)
    model = Tiny()
    cfg = _cfg(rank=5, alpha=10)
    ctx = AdapterContext(exp_id="t3b", components=["O", "G"], rank=5, gralora_k=2)
    gora_alloc.allocate_ranks(cfg, model, None, _loader())
    peft_cfg = GraloraConfig(r=6, gralora_k=2, alpha=10, target_modules=list(TARGETS))
    gora_alloc.patch_config(cfg, peft_cfg, ctx)
    assert peft_cfg.r % 2 == 0
    assert peft_cfg.r == 2 * int(math.floor(5 / 2 + 0.5))  # nearest multiple of k to the budget
    S = (8 + 16) + (12 + 8)  # (m+n) of q_proj and v_proj
    assert ctx.rank_budget_deviation == S * peft_cfg.r - S * 5
    wrapped = get_peft_model(Tiny(), peft_cfg)
    assert tuple(wrapped.base_model.model.q_proj.gralora_A["default"].shape) == (2, 8, peft_cfg.r)


# ---------------------------------------------------------------- T4: X3, k=1 == full matrix

def test_t4_gralora_k1_matches_full_matrix():
    torch.manual_seed(0)
    m, n, r = 10, 12, 4
    gcfg = GraloraConfig(r=r, gralora_k=1, alpha=8, target_modules=["q_proj"])
    peft_model = get_peft_model(One(), gcfg)
    layer = peft_model.base_model.model.q_proj
    A0 = torch.randn(12, r, generator=torch.Generator().manual_seed(4)) * 0.2  # stored (in_blk, r)
    layer.gralora_A["default"].data.copy_(A0)
    G = torch.randn(m, n, generator=torch.Generator().manual_seed(5))
    gora_alloc._G_CACHE.clear()
    gora_alloc._G_CACHE["q_proj"] = G
    cfg = _cfg(components=["Oi"], rank=r, alpha=8, gora_gamma=0.03)
    ctx = AdapterContext(exp_id="t4", components=["Oi"], rank=r, gralora_k=1)
    gora_init.init_adapter(cfg, peft_model, ctx)

    # independent full-matrix reference (spec 3.4 formula, fp64)
    M = A0.T.double()  # (r, n)
    H = M @ M.T
    H = H + 1e-6 * H.diagonal().mean() * torch.eye(r, dtype=H.dtype)
    xi = 0.03 * math.sqrt(n) / ((8 / r) * math.sqrt(r))  # s = alpha/(r+hybrid_r) = 2
    B0_ref = -(torch.linalg.solve(H, (G.double() @ M.T).T)).T * xi  # (m, r)
    got = layer.gralora_B["default"][0].double()  # (r, out_blk) = (r, m)
    assert torch.allclose(got, B0_ref.T, rtol=1e-3, atol=1e-7), (got - B0_ref.T).abs().max()
    assert torch.allclose(layer.gralora_A["default"].double(), A0.double()), "A must stay untouched"


# ---------------------------------------------------------------- T5: shared cache O -> Oi

def test_t5_cache_shared_no_recompute():
    torch.manual_seed(0)
    model = Tiny()
    cfg = _cfg(rank=8, alpha=16)
    p0 = gora_alloc._PASS_COUNT
    ranks = gora_alloc.allocate_ranks(cfg, model, None, _loader())
    assert gora_alloc._PASS_COUNT == p0 + 1

    ctx = AdapterContext(exp_id="t5", components=["O", "L", "Oi"], rank=8)
    ctx.layer_ranks = ranks
    peft_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=list(TARGETS))
    gora_alloc.patch_config(cfg, peft_cfg, ctx)
    wrapped = get_peft_model(Tiny(), peft_cfg)

    ctx2 = AdapterContext(exp_id="t5", components=["O", "L", "Oi"], rank=8)
    ctx2.probe_loader = _loader()  # present but must NOT be used (cache hit)
    gora_init.init_adapter(cfg, wrapped, ctx2)
    assert gora_alloc._PASS_COUNT == p0 + 1, "Oi must reuse the O cache, not re-run the probe"
    assert ctx2.init_mode == "gora"
    assert set(ctx2.gora_grads) == set(TARGETS)
    assert wrapped.base_model.model.q_proj.lora_B["default"].weight.abs().sum() > 0


def test_t5b_oi_standalone_runs_own_probe():
    torch.manual_seed(0)
    gora_alloc._G_CACHE.clear()
    peft_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=list(TARGETS))
    wrapped = get_peft_model(Tiny(), peft_cfg)
    cfg = _cfg(components=["Oi"], rank=8, alpha=16, gora_n_batches=2)
    ctx = AdapterContext(exp_id="t5b", components=["Oi"], rank=8)
    ctx.probe_loader = _loader()
    p0 = gora_alloc._PASS_COUNT
    gora_init.init_adapter(cfg, wrapped, ctx)
    assert gora_alloc._PASS_COUNT == p0 + 1  # exactly one own probe pass
    assert set(gora_alloc.get_cached_grads()) == set(TARGETS)
    assert ctx.init_mode == "gora"
