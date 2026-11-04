"""BA-LoRA output-space regularizers (spec/balora.md) — codes "B", "Bp", "Br".

All three codes share one loss_terms; behaviour differs only through ctx.init_mode:
  - init_mode == "pissa"  -> reference model = snapshot-swap of the PISSA-INIT adapter
    params (APPROVED deviation, docs/decisions.md D-002.9): instead of a frozen W0 copy
    (+1-3 GB) we snapshot (A0, B0) once, then for the reference forward temporarily swap
    the adapter params back to their init values. Full-model forward then computes
    W_res + s*A0*B0 = W0 exactly -> mathematically identical to the paper's
    "pre-trained model". Implementation deviation noted, math equivalence exact.
  - otherwise (default / "gora") -> base_forward() (= adapters disabled) IS W0
    (GoRA init never touches W0, X1) -> zero extra memory.

Loss = L_task + sum(terms), all terms are penalties with PLUS sign (we minimize):
  consistency : KL(P_pt || P_ft) on final logits, mean over response tokens (eq 9),
                scaled by tau^2 with softmax(logits/tau) (U4); NLU variant: squared
                L2(F_p/||F_p|| - F_f/||F_f||) (eq 4, tensor-wise norm U9).
  diversity   : -H(P_ft) full vocab (eq 10; top-k renormalized proxy if
                cfg.balora_top_k_entropy > 0, U5); NLU: off-diagonal covariance
                squared, sum/D (eq 5-6).
  svdr        : -(sum_{i<=k} sigma~_i) / (sum_all sigma~_i) of the batch-pooled
                non-pad logits matrix Z (M x V) (eq 11; denominator = sum of
                singular values per D-002.2, NOT Frobenius). NLG: chunked randomized
                SVD (U10: p=5, niter=1); NLU: exact SVD, k=5.

Schedule "spec42" (spec/balora.md section 4.2; NOT in paper, approved deviation):
  cosine warmup->decay for lambda1; two_phase (warmup->hold->ramp-down) for lambda2/3;
  warmup 0.2, ramp 0.05. Factors applied INSIDE the returned term values.

Timing (static estimate, Qwen2.5-0.5B, M=4096, V=152k, T4; measure for real at stage 5
into docs/budget.md): consistency adds 1 no-grad reference forward; svdr ~5-15 ms
(randomized, q=k+5); diversity ~few ms. Total +25-40% per step (spec section 6).

fp16/T4 rules (spec section 7): all reg math in explicit fp32 islands (autocast off),
log-space softmax for KL, SVD in fp32, target fully detached, chunked over M rows
(CHUNK=1024 rows ~ 0.6 GB fp32 at V=152k), guard denominators against 0.

Ordering invariant: the reference forward (with param swap for pissa) runs BEFORE the
student forward of this call — pipeline backwards each micro-batch right after
loss_terms returns, so no autograd graph is ever live while params are swapped.
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F

from ..registry import register

CHUNK = 1024  # rows per fp32 chunk (spec section 7.4)


def _ckpt(fn, *args):
    """Gradient-checkpoint a chunk kernel: fp32 softmax/matmul intermediates of
    (CHUNK=1024 rows x V=152k) ~0.6 GiB each are RECOMPUTED in backward instead of
    retained for the whole micro-batch (Bẫy 11: Bp OOM'd at 13.4/14.6 GiB on T4).
    Falls back to a plain call when nothing needs grad (eval / fake models).
    """
    if not any(torch.is_tensor(a) and a.requires_grad for a in args):
        return fn(*args)
    return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)

# output dict keys (task contract): "consistency" / "diversity" / "svdr"


def _schedule_factors(cfg, ctx) -> tuple[float, float, float]:
    """Return multiplicative factors (f1, f2, f3) for (lambda1, lambda2, lambda3).

    spec/balora.md section 4.2 with w = warmup_ratio, q = ramp_up_ratio, u = t/T:
      lambda1(u) = u/w                        if u <= w      (linear warmup)
                 = 0.5(1+cos(pi(u-w)/(1-w)))  otherwise      (cosine decay to 0)
      lambda23(u) = u/w                       if u <= w
                  = 1                         if w < u <= 1-q
                  = (1-u)/q                   if u > 1-q
    Checks: lambda(0)=0, lambda(w)=lambda, lambda(1-q)=lambda, lambda(1)=0.
    Any other cfg.balora_schedule (e.g. "none") -> factors 1.0 (raw paper lambdas).
    """
    if getattr(cfg, "balora_schedule", "none") != "spec42" or ctx.total_steps <= 0:
        return 1.0, 1.0, 1.0
    u = min(max(ctx.step / ctx.total_steps, 0.0), 1.0)
    w = float(cfg.balora_warmup_ratio)
    q = float(cfg.balora_ramp_up_ratio)
    if u <= w and w > 0:
        f1 = f23 = u / w
    else:
        f1 = 0.5 * (1.0 + math.cos(math.pi * (u - w) / (1.0 - w)))
        f23 = 1.0 if u <= 1.0 - q else (1.0 - u) / q
    return f1, f23, f23


def _attn_mask(batch: dict, logits: torch.Tensor) -> torch.Tensor:
    am = batch.get("attention_mask")
    if am is None:
        return torch.ones(logits.shape[:-1], dtype=torch.bool, device=logits.device)
    return am.bool()


def _cr_mask(batch: dict, logits: torch.Tensor) -> torch.Tensor:
    """Response-only mask (U7 as approved): labels != -100, fallback attention mask."""
    labels = batch.get("labels")
    m = _attn_mask(batch, logits)
    if labels is not None:
        m = (labels != -100) & m
    return m


def _pooled_rows(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Select non-masked positions -> (M, V) matrix (U6: one matrix per batch)."""
    return logits[mask]


def _snapshot_adapter_params(model, ctx) -> None:
    """First-call snapshot of all trainable (= adapter) params, detached clones.

    For init_mode == "pissa" these are the PISSA-INIT values (A0, B0). Cached in
    ctx.hooks["_pissa_snapshot"]; cheap (rank*dim floats per layer, ~MB total).
    """
    if "_pissa_snapshot" in ctx.hooks:
        return
    snap = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    ctx.hooks["_pissa_snapshot"] = snap


def _w0_forward(model, ctx, batch) -> torch.Tensor:
    """Reference forward for Bp: temporarily restore PISSA-INIT adapter params so the
    full model computes W0 = W_res + s*A0*B0 (paper's "pre-trained model"), then swap
    the current params back. Must be called before any student forward of this call
    (no live autograd graph during the swap -> in-place restore is bitwise exact and
    cannot corrupt saved tensors). Implementation deviation from "keep a frozen W0
    copy"; mathematically equivalent.
    """
    snap: dict = ctx.hooks["_pissa_snapshot"]
    saved: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in snap:
                saved[name] = p.detach().clone()
                p.copy_(snap[name])
        try:
            out = model(**batch)
        finally:  # always restore, even if forward raises
            for name, p in model.named_parameters():
                if name in saved:
                    p.copy_(saved[name])
    return out.logits.detach()


def _cr_chunk(rows_chunk: torch.Tensor, lp_pt: torch.Tensor, p_pt: torch.Tensor, tau: float) -> torch.Tensor:
    lp_ft = F.log_softmax(rows_chunk.float() / tau, dim=-1)
    return (p_pt * (lp_pt - lp_ft)).sum()


def _cr_nlg(logits_ft: torch.Tensor, logits_ref: torch.Tensor, mask, tau: float) -> torch.Tensor:
    """KL(P_pt || P_ft) per eq 9, response tokens only, chunked fp32 (checkpointed),
    scaled by tau^2 (U4)."""
    rows_ft = logits_ft[mask]  # (M, V), keeps grad
    rows_pt = logits_ref[mask].detach()
    total = torch.zeros((), dtype=torch.float32, device=logits_ft.device)
    count = 0
    for i in range(0, rows_ft.shape[0], CHUNK):
        chunk = rows_ft[i : i + CHUNK]
        with torch.no_grad():  # teacher probs are constants -> computed once, not checkpointed
            lp_pt = F.log_softmax(rows_pt[i : i + CHUNK].float() / tau, dim=-1)
            p_pt = lp_pt.exp()
        total = total + _ckpt(_cr_chunk, chunk, lp_pt, p_pt, tau)
        count += chunk.shape[0]
    return tau * tau * total / max(count, 1)


def _dr_chunk(rows_chunk: torch.Tensor, top_k: int) -> torch.Tensor:
    z = rows_chunk.float()
    if top_k and top_k > 0:
        z = z.topk(min(top_k, z.shape[-1]), dim=-1).values
    p = torch.softmax(z, dim=-1)
    # entr(x) = -x*ln(x) with proper 0 handling (0*log0 = 0, no NaN)
    return torch.special.entr(p).sum()


def _dr_nlg(logits_ft: torch.Tensor, mask, top_k: int) -> torch.Tensor:
    """-H(P_ft) per eq 10 (full vocab; top-k renormalized proxy if top_k > 0), mean over
    valid tokens, chunked + checkpointed fp32."""
    rows = logits_ft[mask]
    total = torch.zeros((), dtype=torch.float32, device=logits_ft.device)
    count = 0
    for i in range(0, rows.shape[0], CHUNK):
        chunk = rows[i : i + CHUNK]
        total = total + _ckpt(_dr_chunk, chunk, top_k)
        count += chunk.shape[0]
    return -total / max(count, 1)


def _chunks(n: int):
    return range(0, n, CHUNK)


def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.float() @ b


def _mm_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.float().T @ b.float()


def _svdr_rand(rows: torch.Tensor, k: int, p_over: int = 5, niter: int = 1) -> torch.Tensor:
    """Randomized-SVD top-k mass ratio (U10: p=5, niter=1), chunked (spec section 3.1).

    Chunk matmuls are gradient-checkpointed: the fp32 casts of a (CHUNK x V) slice are
    ~0.6 GiB each and the micro-batch ran out of memory holding them (Bẫy 11).

    Note: svdvals backward has NaN gradients for exactly-repeated singular values
    (measure-zero for real logits, which are full-rank); forward values are always safe.
    """
    M, V = rows.shape
    q = min(k + p_over, min(M, V))
    if M == 0 or q <= 0:
        return torch.zeros((), dtype=torch.float32, device=rows.device)
    dev = rows.device
    G = torch.randn(V, q, device=dev, dtype=torch.float32)  # outside checkpoint: no RNG replay
    # Y = Z @ G (M, q), chunked over rows, autograd preserved per chunk
    Y = torch.cat([_ckpt(_mm, rows[i : i + CHUNK], G) for i in _chunks(M)], dim=0)
    for _ in range(niter):
        # power iteration Y <- Z @ (Z^T @ Y); both products chunked over M rows
        ZtY = torch.zeros(V, q, device=dev, dtype=torch.float32)
        for i in _chunks(M):
            ZtY = ZtY + _ckpt(_mm_t, rows[i : i + CHUNK], Y[i : i + CHUNK])
        Y = torch.cat([_ckpt(_mm, rows[i : i + CHUNK], ZtY) for i in _chunks(M)], dim=0)  # (M, q)
    Q, _ = torch.linalg.qr(Y)  # (M, q) orthonormal
    B = torch.cat([_ckpt(_mm_t, Q[i : i + CHUNK], rows[i : i + CHUNK])
                   for i in _chunks(M)], dim=1)  # (q, V)
    s = torch.linalg.svdvals(B)  # top-q approx singular values of Z, descending
    k = min(k, s.numel())
    denom = s.sum()
    if float(denom.detach()) == 0.0:
        return torch.zeros((), dtype=torch.float32, device=dev)
    return -(s[:k].sum() / denom)


def _svdr_exact(rows: torch.Tensor, k: int) -> torch.Tensor:
    """Exact-SVD top-k mass ratio (NLU / small matrices; eq 7)."""
    if rows.shape[0] == 0:
        return torch.zeros((), dtype=torch.float32, device=rows.device)
    s = torch.linalg.svdvals(rows.float())
    k = min(k, s.numel())
    denom = s.sum()
    if float(denom.detach()) == 0.0:
        return torch.zeros((), dtype=torch.float32, device=rows.device)
    return -(s[:k].sum() / denom)


def _cr_nlu(z_ft: torch.Tensor, z_pt: torch.Tensor) -> torch.Tensor:
    """eq 4: squared L2 distance between tensor-wise normalized logits (U9)."""
    with torch.no_grad():
        n_pt = z_pt.float() / z_pt.float().norm().clamp_min(1e-12)
    n_ft = z_ft.float() / z_ft.float().norm().clamp_min(1e-12)
    return (n_ft - n_pt).pow(2).sum()


def _dr_nlu(z: torch.Tensor) -> torch.Tensor:
    """eq 5-6: off-diagonal covariance squared, summed over i!=j, divided by D."""
    M, D = z.shape
    if M < 2 or D < 2:
        return torch.zeros((), dtype=torch.float32, device=z.device)
    zc = z.float() - z.float().mean(dim=0, keepdim=True)
    C = (zc.T @ zc) / (M - 1)
    off = C - torch.diag(torch.diagonal(C))
    return off.pow(2).sum() / D


def _nlu_rows(logits: torch.Tensor, batch: dict) -> torch.Tensor:
    """NLU sampling matrix: 2D logits (B, C) used as-is; 3D pooled over non-pad positions."""
    if logits.dim() == 2:
        return logits
    return logits[_attn_mask(batch, logits)]


def loss_terms(cfg, model, ctx, batch, base_forward, logits_ft=None):
    """Hook contract: loss_terms(cfg, model, ctx, batch, base_forward, logits_ft=None)
    -> dict[str, Tensor].

    Returns only terms whose scheduled lambda > 0; values already include lambda(t)
    (pipeline just does total = task_loss + sum(terms.values())). All computation in
    fp32 islands (autocast disabled) per spec section 7.

    logits_ft: the pipeline's task-loss forward logits. Passing them avoids a second
    student forward of the same batch (~1.2 GiB fp16 logits + activation graph at
    0.5B/8x512/V=152k) — that duplicate forward caused the Bp OOM (Bẫy 11). None
    (tests, standalone use) -> this function runs its own forward.
    """
    amp_off = (torch.autocast("cuda", enabled=False) if torch.cuda.is_available()
               else contextlib.nullcontext())
    with amp_off:
        variant = getattr(cfg, "balora_variant", "nlg")
        f1, f2, f3 = _schedule_factors(cfg, ctx)
        lam1 = float(cfg.balora_lambda1) * f1
        lam2 = float(cfg.balora_lambda2) * f2
        lam3 = float(cfg.balora_lambda3) * f3

        # Bp: snapshot PISSA-INIT params on the very first call (cheap), even if CR
        # is currently off, so the reference is pinned before any training step.
        if ctx.init_mode == "pissa":
            _snapshot_adapter_params(model, ctx)

        if lam1 <= 0 and lam2 <= 0 and lam3 <= 0:
            return {}  # skip ref + student forwards entirely (no +50% step cost)

        # reference forward FIRST: for pissa it swaps adapter params, which is only
        # safe while no autograd graph is live (pipeline backwards after this returns)
        ref = None
        if lam1 > 0:
            if ctx.init_mode == "pissa":
                ref = _w0_forward(model, ctx, batch)
            elif base_forward is not None:
                ref = base_forward(batch).logits.detach()
            else:
                with model.disable_adapter():
                    ref = model(**batch).logits.detach()

        if logits_ft is None:  # no task-loss forward available (tests/standalone) -> run one
            logits_ft = model(**batch).logits

        terms: dict[str, torch.Tensor] = {}
        if variant == "nlu":
            z_ft = _nlu_rows(logits_ft, batch)
            k = 5  # NLU fixed per paper section 4.2 (user D-002.2), ignores balora_svd_k
            if lam1 > 0:
                terms["consistency"] = lam1 * _cr_nlu(z_ft, _nlu_rows(ref, batch))
            if lam2 > 0:
                terms["diversity"] = lam2 * _dr_nlu(z_ft)
            if lam3 > 0:
                terms["svdr"] = lam3 * _svdr_exact(z_ft, k)
            return terms

        # NLG (default)
        mask_cr = _cr_mask(batch, logits_ft)
        mask_tok = _attn_mask(batch, logits_ft)
        if lam1 > 0:
            tau = float(cfg.balora_distill_temp)
            terms["consistency"] = lam1 * _cr_nlg(logits_ft, ref, mask_cr, tau)
        if lam2 > 0:
            terms["diversity"] = lam2 * _dr_nlg(logits_ft, mask_tok, int(cfg.balora_top_k_entropy))
        if lam3 > 0:
            terms["svdr"] = lam3 * _svdr_rand(_pooled_rows(logits_ft, mask_tok), int(cfg.balora_svd_k))
        return terms


register("B", loss_terms=loss_terms)
register("Bp", loss_terms=loss_terms)
register("Br", loss_terms=loss_terms)
