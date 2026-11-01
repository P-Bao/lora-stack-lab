"""GoRA component `O` — gradient-driven rank allocation (clean-room from spec/gora.md).

Phase (a) [paper §3.2 / Algorithm 1 l.4–10]: one streaming pass over N batches on the BASE
model; per-batch grads offloaded to CPU fp32 buffers via param hooks, no optimizer step.
Model state during probe: base model (pipeline runs O before get_peft_model → no adapter
ambiguity), train() mode + fixed seed (spec 9.4). fp32 accumulation, no autocast (spec 7.1);
any global loss scale cancels in the normalized importance (spec 3.1).

Phase (b): I(W)=avg(|W⊙G|) fp32 [Eq.(5)], advantage A_i=I_i/ΣI [Eq.(6)],
budget B_tot = Σ(m+n)·r_ref [Eq.(7), (m+n) form per decisions.md D-002.5],
r_i = clip(floor(B_tot·A_i/(m+n)), r_min, r_max) [Eq.(8), FLOOR per black-box V2-C/D + D-003] with r_min=r_ref/2, r_max=4·r_ref
[paper B.3]; one deterministic budget-rescale round when clamp drift > 10% (spec 9.7 / D-002.6).
"""
from __future__ import annotations

import math
import re
import warnings

import torch
from torch import nn
from peft import GraloraConfig, LoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer

from ..config import ExpConfig
from ..registry import register
from ..utils.seed import set_seed

_PREFIX = "base_model.model."
_G_CACHE: dict[str, torch.Tensor] = {}  # base layer name -> G (CPU fp32), shared O <-> Oi
_LAYER_SHAPES: dict[str, tuple[int, int]] = {}  # base layer name -> (m, n) = (out, in)
_PASS_COUNT = 0  # number of gradient probe passes (observability for tests)


def get_cached_grads() -> dict[str, torch.Tensor]:
    """Shared G cache filled by O (allocate_ranks), consumed by Oi (gora_init)."""
    return _G_CACHE


def _strip_prefix(name: str) -> str:
    return name[len(_PREFIX):] if name.startswith(_PREFIX) else name


def _name_matches(name: str, targets) -> bool:
    if targets is None:
        return False
    if isinstance(targets, str):
        return re.fullmatch(targets, name) is not None
    return any(name == t or name.endswith("." + t) for t in targets)


def _iter_target_weights(model, target_modules=None):
    """(base_layer_name, weight) pairs. BASE model: suffix-match cfg.target_modules on
    nn.Linear; PEFT-wrapped model (Oi standalone probe): every BaseTunerLayer base weight."""
    for name, mod in model.named_modules():
        if isinstance(mod, BaseTunerLayer):
            yield _strip_prefix(name), mod.get_base_layer().weight
        elif isinstance(mod, nn.Linear) and _name_matches(name, target_modules):
            yield name, mod.weight


def _accumulate_grads(cfg, model, probe_loader) -> dict[str, torch.Tensor]:
    """Phase (a): G = mean dL/dW over min(len(loader), cfg.gora_n_batches) batches.
    CPU fp32 buffers; hooks removed, requires_grad restored, grads cleared (spec 7.5)."""
    global _PASS_COUNT
    targets = dict(_iter_target_weights(model, getattr(cfg, "target_modules", None)))
    if not targets:
        raise ValueError("GoRA probe: no target layers found (check cfg.target_modules)")
    G = {name: torch.zeros_like(p, dtype=torch.float32, device="cpu") for name, p in targets.items()}
    handles, saved = [], []
    # freeze non-targets during the probe: backward otherwise builds grads for the whole
    # model (~+1 GB/0.5B transient at build time — reviewer 🟡)
    all_saved = [(p, p.requires_grad) for p in model.parameters()]
    target_ids = {id(p) for _, p in targets.items()}
    device = next(model.parameters()).device
    was_training = model.training
    n = 0
    try:
        for name, p in targets.items():
            saved.append((p, p.requires_grad))
            p.requires_grad_(True)

            def _hook(grad, _name=name):
                G[_name] += grad.detach().float().cpu()  # offload per batch (Algorithm 1 l.10)

            handles.append(p.register_hook(_hook))
        for p, rg in all_saved:
            if id(p) not in target_ids:
                p.requires_grad_(False)
        model.train()  # spec 9.4: train mode; seed fixed by caller (allocate_ranks / pipeline)
        for batch in probe_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()  # fp32 loss, no autocast (spec 7.1)
            model.zero_grad(set_to_none=False)  # Algorithm 1: accumulate, no optimizer step
            n += 1
            if n >= cfg.gora_n_batches:
                break
    finally:
        for h in handles:
            h.remove()
        for p, rg in all_saved:
            p.requires_grad_(rg)
        model.zero_grad(set_to_none=True)  # spec 7.5: no stray grads for the training optimizer
        model.train(was_training)
    _PASS_COUNT += 1
    if n == 0:
        raise ValueError("GoRA probe: loader yielded no batches")
    return {name: g / n for name, g in G.items()}


def _allocate_ranks_from_importance(importance: dict[str, float],
                                    shapes: dict[str, tuple[int, int]],
                                    cfg) -> dict[str, int]:
    """Pure phase (b): Eq.(5)-(8) + clamp + one rescale round (spec 9.7 / D-002.6)."""
    r_ref = cfg.rank
    r_min = cfg.gora_r_min if cfg.gora_r_min is not None else max(1, r_ref // 2)  # [paper B.3]
    r_max = cfg.gora_r_max if cfg.gora_r_max is not None else 4 * r_ref           # [paper B.3]
    total = sum(importance.values())
    if total <= 0.0:  # degenerate all-zero G: uniform advantage keeps the budget exact
        advantage = {k: 1.0 / len(importance) for k in importance}
    else:
        advantage = {k: v / total for k, v in importance.items()}  # Eq.(6)
    S = sum(m + n for m, n in shapes.values())
    B_tot = S * r_ref  # Eq.(7), (m+n) form (D-002.5)

    def _round_clip(budget: float) -> dict[str, int]:
        out = {}
        for name, (m, n) in shapes.items():
            raw = budget * advantage[name] / (m + n)
            out[name] = int(min(r_max, max(r_min, math.floor(raw))))  # floor — matches author behavior (black-box V2-C/D, D-003)
        return out

    ranks = _round_clip(B_tot)
    P_hat = sum((shapes[k][0] + shapes[k][1]) * r for k, r in ranks.items())
    if abs(P_hat - B_tot) > 0.10 * B_tot:
        # one deterministic rescale round (spec 9.7 / D-002.6); residual drift only warned
        ranks = _round_clip(B_tot * B_tot / P_hat)
        P_hat2 = sum((shapes[k][0] + shapes[k][1]) * r for k, r in ranks.items())
        if abs(P_hat2 - B_tot) > 0.10 * B_tot:
            warnings.warn(f"GoRA rank budget drift after rescale: {abs(P_hat2 - B_tot) / B_tot:.1%}")
    return ranks


def allocate_ranks(cfg, model, tokenizer, probe_loader) -> dict[str, int]:
    """Hook O (before get_peft_model). Returns {full module name, no peft prefix: rank}."""
    set_seed(cfg.seed)  # spec 4 phase (a): deterministic probe (plan.md rule 5)
    G = _accumulate_grads(cfg, model, probe_loader)
    _G_CACHE.clear()
    _G_CACHE.update(G)
    weights = dict(_iter_target_weights(model, cfg.target_modules))
    shapes, importance = {}, {}
    for name, g in G.items():
        w = weights[name].detach().float().cpu()
        shapes[name] = (int(w.shape[0]), int(w.shape[1]))  # (m, n) = (out, in)
        importance[name] = float((w * g).abs().mean())     # Eq.(5), fp32
    ranks = _allocate_ranks_from_importance(importance, shapes, cfg)
    _LAYER_SHAPES.clear()
    _LAYER_SHAPES.update(shapes)  # consumed by patch_config (GraloraConfig global-r path)
    return ranks


def patch_config(cfg, peft_config, ctx) -> None:
    """Hook O (after peft_config build, before get_peft_model).
    LoraConfig: rank_pattern per layer + alpha_pattern keeping per-layer effective scaling
    s = alpha/r_ref (so L vs L+O differ only in rank allocation, not adapter scale).
    GraloraConfig: single global r nearest the budget, multiple of gralora_k (D-X2); param
    drift vs the LoRA budget recorded in ctx.rank_budget_deviation (signed)."""
    if isinstance(peft_config, GraloraConfig):
        if not _LAYER_SHAPES:
            raise ValueError("GoRA patch_config: allocate_ranks must run first (needs layer shapes)")
        k = peft_config.gralora_k
        S = sum(m + n for m, n in _LAYER_SHAPES.values())
        B_tot = S * cfg.rank
        r_global = max(k, k * int(math.floor(B_tot / (k * S) + 0.5)))  # nearest multiple of k
        peft_config.r = int(r_global)
        ctx.rank_budget_deviation = int(S * r_global - B_tot)
        return
    ranks = ctx.layer_ranks or {}
    if not ranks:
        return
    # D-002.1: alpha/r for ALL configs; gora_use_rslora is an ablation flag only (user 2026-11-01).
    if cfg.gora_use_rslora:
        peft_config.use_rslora = True
        s_ref = cfg.alpha / math.sqrt(cfg.rank)
    else:
        s_ref = cfg.alpha / cfg.rank
    for name, r in ranks.items():
        # NOTE 9.11 (Kaggle smoke pending): PEFT matches pattern keys as suffix regex
        # (.*\.)?(key)$ — keys stored re.escape()d so dots stay literal (digits after a dot
        # in neighbour layer names can never alias). Fallback if matching fails on Kaggle:
        # patch per-layer rank manually after build.
        key = re.escape(name)
        peft_config.rank_pattern[key] = int(r)
        peft_config.alpha_pattern[key] = s_ref * (math.sqrt(r) if cfg.gora_use_rslora else float(r))


register("O", allocate_ranks=allocate_ranks, patch_config=patch_config)
