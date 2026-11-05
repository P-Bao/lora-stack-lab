"""GoRA component `Oi` — gradient-driven B0 init (clean-room from spec/gora.md §3.4–3.5).

Mapping (spec 2.1, PEFT ΔW = s·lora_B·lora_A): keep lora_A = PEFT-native kaiming ("A₀ kept",
PEFT-native per D-002.7), compute lora_B:
    lora_B0 = -xi · G · Mᵀ (M Mᵀ + eps·I)⁻¹,  M = lora_A (r×n), eps_rel = 1e-6 (spec 9.12)
=> ΔW_init = s·lora_B0·M = -s·xi·G·P_row ≈ -gamma·G (one SGD-like step, [paper Eq.(10)]).
xi (PEFT-native column of the spec 3.5 table) = gamma·√n/(s·√r); with s = α/r this equals
gamma·√(r·n)/α, with s = α/√r it equals gamma·√n/α. s is read from the BUILT adapter
(spec 6.4: never hardcode — covers rslora ablation and GraLoRA α/(r+hybrid_r)).
G comes from the gora_alloc cache (shared O → Oi, no recompute); if empty (Oi standalone)
run the probe pass here via ctx.probe_loader (spec 6.2).
GraLoRA (X3): assign per slice. Block (n2, n1) of ΔW equals B[n2] rows [n1·rk,(n1+1)·rk) @
(A[n1] cols [n2·rk,(n2+1)·rk])ᵀ with rk = r/k, so each block gets its own projection init from
its own G block; k=1 degenerates exactly to the full-matrix formula (tested).
fp16/T4 (spec 7): all linear algebra fp32 + ridge, no eigvalsh; params keep their dtype.
"""
from __future__ import annotations

import math

import torch
from peft.tuners.tuners_utils import BaseTunerLayer

from ..registry import register
from . import gora_alloc

_EPS_REL = 1e-6
_GAMMA_TABLE = {8: 5e-2, 32: 1e-2, 128: 5e-3}  # [paper B.3 / Table 9], indexed by r_ref


def _resolve_gamma(cfg) -> float:
    """cfg.gora_gamma wins; else paper table; else linear interpolation in 1/√r_ref (spec 9.3)."""
    if cfg.gora_gamma is not None:
        return float(cfg.gora_gamma)
    if cfg.rank in _GAMMA_TABLE:
        return _GAMMA_TABLE[cfg.rank]
    x = 1.0 / math.sqrt(cfg.rank)
    pts = sorted((1.0 / math.sqrt(r), g) for r, g in _GAMMA_TABLE.items())  # x asc, gamma desc
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return pts[-1][1]


def _b0_solve(G: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """B0 = -G Mᵀ (M Mᵀ + eps_rel·I)⁻¹ in fp32 (spec 7.2); ridge mandatory (spec 7.3/9.12)."""
    H = M @ M.T  # (r, r) fp32 — the inverse is never n×n (spec 3.4)
    r = M.shape[0]
    H = H + _EPS_REL * H.diagonal().mean() * torch.eye(r, dtype=H.dtype, device=H.device)
    R = G @ M.T  # (m, r)
    try:
        return -torch.linalg.solve(H, R.T).T  # (m, r)
    except Exception:
        return -torch.linalg.lstsq(H, R.T).solution.T  # ridge keeps H pd; belt-and-braces


def _xi(gamma: float, scaling: float, r: int, n: int) -> float:
    # spec 3.5 PEFT-native column rewritten with the adapter's actual scaling s (spec 6.4):
    # xi = gamma / (s·√(r/n)) == gamma·√n/(s·√r); s=α/r ⇒ γ√(rn)/α, s=α/√r ⇒ γ√n/α.
    return gamma * math.sqrt(n) / (scaling * math.sqrt(r))


def _adapted_layers(model):
    """[(base_name, module, kind)] for every adapted Linear (kind: 'lora' | 'gralora')."""
    out = []
    for name, mod in model.named_modules():
        if not isinstance(mod, BaseTunerLayer):
            continue
        base = gora_alloc._strip_prefix(name)
        if hasattr(mod, "lora_A"):
            out.append((base, mod, "lora"))
        elif hasattr(mod, "gralora_A"):
            out.append((base, mod, "gralora"))
    return out


def _init_lora(mod, G, gamma) -> None:
    ad = "default"
    a = mod.lora_A[ad].weight  # (r, n), PEFT-native kaiming — untouched
    b = mod.lora_B[ad].weight  # (m, r)
    r, n = a.shape
    Gd = G.to(device=a.device, dtype=torch.float32)  # one layer on GPU at a time (spec 7.4)
    B0 = _b0_solve(Gd, a.detach().float())
    B0 *= _xi(gamma, float(mod.scaling[ad]), int(r), int(n))
    with torch.no_grad():
        b.copy_(B0.to(dtype=b.dtype))  # keep param dtype (fp16/T4 rule)
    del Gd, B0


def _init_gralora(mod, G, gamma) -> None:
    ad = "default"
    A3 = mod.gralora_A[ad]  # (k, in_blk, r) — stored transposed vs LoRA (X3, peft_inventory §3)
    B3 = mod.gralora_B[ad]  # (k, r, out_blk)
    k = int(mod.gralora_k[ad])
    r = int(mod.r[ad])
    n, m = int(mod.in_features), int(mod.out_features)
    if int(mod.hybrid_r[ad]) > 0:
        raise NotImplementedError("GoRA init with GraLoRA hybrid_r > 0 is out of scope (lab uses hybrid_r=0)")
    rk = r // k
    in_blk, out_blk = n // k, m // k
    Gd = G.to(device=A3.device, dtype=torch.float32)
    xi = _xi(gamma, float(mod.scaling[ad]), r, n)
    # Global xi (not per-block) is intentional: with per-block projectors of rank rk on R^in_blk,
    # E‖G_blk·P̃‖/‖G_blk‖ = √(rk/in_blk), and √(n/r)·√(rk/in_blk) = 1, so per-block
    # ‖ΔW_blk‖/‖G_blk‖ ≈ γ still holds; k=1 coincides exactly with the LoRA formula (T4).
    with torch.no_grad():
        for n2 in range(k):
            for n1 in range(k):
                # M̃ = A block n1, rank columns owned by output strip n2 → (rk, in_blk)
                Mt = A3[n1][:, n2 * rk:(n2 + 1) * rk].detach().float().T
                Gb = Gd[n2 * out_blk:(n2 + 1) * out_blk, n1 * in_blk:(n1 + 1) * in_blk]
                Bb = _b0_solve(Gb, Mt) * xi  # (out_blk, rk)
                B3[n2][n1 * rk:(n1 + 1) * rk, :].copy_(Bb.T.to(dtype=B3.dtype))
    del Gd


def init_adapter(cfg, model, ctx) -> None:
    """Hook Oi (after get_peft_model). Applies phase (c); sets ctx.init_mode = 'gora' (spec 6.2)."""
    gamma = _resolve_gamma(cfg)
    grads = gora_alloc.get_cached_grads()  # shared O → Oi cache — no recompute
    need = _adapted_layers(model)
    if not need:
        raise ValueError("GoRA init: no adapted layers found (run after get_peft_model)")
    missing = [name for name, _, _ in need if name not in grads]
    if missing:
        if ctx.probe_loader is None:
            head = ", ".join(missing[:3]) + ("..." if len(missing) > 3 else "")
            raise ValueError(f"GoRA init: no cached G for {head} and ctx.probe_loader is None")
        G = gora_alloc._accumulate_grads(cfg, model, ctx.probe_loader)  # Oi standalone (spec 6.2)
        gora_alloc._G_CACHE.clear()
        gora_alloc._G_CACHE.update(G)
        grads = G
    ctx.gora_grads.update(grads)  # mirror into ctx (contract field: gora_grads, CPU fp32)
    for base_name, mod, kind in need:
        G = grads[base_name]
        if kind == "lora":
            _init_lora(mod, G, gamma)
        else:
            _init_gralora(mod, G, gamma)
    ctx.init_mode = "gora"
    # TODO(D-002.1): user pinned alpha/r for ALL configs; cfg.gora_use_rslora stays an ablation
    # flag. xi reads the adapter's actual scaling (spec 6.4), so an rslora ablation run needs no
    # code change here (xi becomes gamma*sqrt(n)/alpha automatically). Nothing else pending.


register("Oi", init_adapter=init_adapter)
