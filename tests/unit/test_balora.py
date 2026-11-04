"""Unit tests for lorastack.losses.balora_reg (spec/balora.md, task T1-T6 + NLU sanity).

CPU only, tiny fake models, no version asserts. Runs under pytest OR standalone:
    pytest tests/unit/test_balora.py
    python tests/unit/run_balora_checks.py    (plain-assert fallback, no pytest needed)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lorastack.adapters.base import AdapterContext  # noqa: E402
from lorastack.config import ExpConfig  # noqa: E402
from lorastack.losses import balora_reg as br  # noqa: E402

torch.manual_seed(0)


def _v(t) -> float:  # read a (possibly differentiable) scalar loss value
    return float(t.detach())


def _cfg(**kw) -> ExpConfig:
    base = dict(components=["Br"], balora_schedule="none", balora_variant="nlg",
                balora_lambda1=0.0, balora_lambda2=0.0, balora_lambda3=0.0,
                balora_distill_temp=2.0, balora_top_k_entropy=0, balora_svd_k=10)
    base.update(kw)
    return ExpConfig(**base)


def _ctx(**kw) -> AdapterContext:
    base = dict(exp_id="t", components=["Br"], rank=2)
    base.update(kw)
    return AdapterContext(**base)


def _batch(B=1, L=1, V=8, labels=None) -> dict:
    b = {"input_ids": torch.randint(0, V, (B, L)),
         "attention_mask": torch.ones(B, L, dtype=torch.long)}
    if labels is not None:
        b["labels"] = labels
    return b


def _no_base(*a, **k):
    raise AssertionError("base_forward must not be called")


class FixedLogits(torch.nn.Module):
    """Returns a fixed (trainable) (1, L, V) logits tensor expanded over batch."""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone().reshape(1, -1, logits.shape[-1]))

    def forward(self, input_ids, attention_mask=None, labels=None, **kw):
        B = input_ids.shape[0]
        return SimpleNamespace(logits=self.logits.expand(B, -1, -1))


class FixedLogits2D(torch.nn.Module):
    """NLU-style model: fixed (B, C) logits."""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())

    def forward(self, input_ids, attention_mask=None, labels=None, **kw):
        return SimpleNamespace(logits=self.logits)


class MiniPissa(torch.nn.Module):
    """Tiny PEFT-like layer with PiSSA semantics: logits = h@W_res.T + s*(h@A.T@B.T),
    with W_res = W0 - s*B0@A0, so at init (A,B)=(A0,B0) the model computes W0 exactly.
    Trainable: lora_A, lora_B only (what _snapshot_adapter_params must capture)."""

    def __init__(self, V=8, d=4, r=2, scale=2.0):
        super().__init__()
        torch.manual_seed(7)
        self.scale = scale
        W0 = torch.randn(V, d)
        A0 = torch.randn(r, d) * 0.5
        B0 = torch.randn(V, r) * 0.5
        self.register_buffer("W0", W0)
        self.embed = torch.nn.Embedding(V, d)
        self.base = torch.nn.Linear(d, V, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(V, d) * 0.3)
            self.base.weight.copy_(W0 - scale * (B0 @ A0))  # W_res
        self.embed.weight.requires_grad_(False)
        self.base.weight.requires_grad_(False)
        self.lora_A = torch.nn.Parameter(A0.clone())
        self.lora_B = torch.nn.Parameter(B0.clone())

    def forward(self, input_ids, attention_mask=None, labels=None, **kw):
        h = self.embed(input_ids)                      # (B, L, d)
        logits = h @ self.base.weight.T + self.scale * (h @ self.lora_A.T @ self.lora_B.T)
        return SimpleNamespace(logits=logits)


# ---------------------------------------------------------------- T1: lambda = 0 -> empty dict

def test_t1_zero_lambda_returns_empty():
    model = FixedLogits(torch.randn(1, 3, 8))
    terms = br.loss_terms(_cfg(), model, _ctx(), _batch(L=3), _no_base)
    assert terms == {}
    # schedule "spec42" at step 0 zeroes every lambda too -> skipped entirely
    ctx = _ctx(step=0, total_steps=100)
    terms = br.loss_terms(_cfg(balora_schedule="spec42", balora_lambda1=1.0,
                               balora_lambda2=1.0, balora_lambda3=1.0), model, ctx,
                          _batch(L=3), _no_base)
    assert terms == {}


# ---------------------------------------------------------------- T2: CR (KL, tau^2, mask)

def test_t2_cr_kl_identical_zero_different_positive_tau_squared():
    tau = 2.0
    stu = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
    ref = torch.tensor([[[3.0, 2.0, 1.0, 0.0]]])

    def base(b):
        return SimpleNamespace(logits=ref.detach().expand(b["input_ids"].shape[0], -1, -1))

    cfg = _cfg(balora_lambda1=1.0, balora_distill_temp=tau)
    # identical distributions -> KL = 0
    same = br.loss_terms(cfg, FixedLogits(ref), _ctx(), _batch(L=1), base)
    assert _v(same["consistency"]) < 1e-7
    # different -> KL > 0, and equals the hand-derived eq9+U4 value:
    # tau^2 * sum_i p_i (ln p_i - ln q_i), p = softmax(ref/tau), q = softmax(stu/tau)
    terms = br.loss_terms(cfg, FixedLogits(stu), _ctx(), _batch(L=1), base)
    p = torch.softmax(ref.reshape(-1) / tau, -1)
    q = torch.softmax(stu.reshape(-1) / tau, -1)
    expected = tau * tau * float((p * (p.log() - q.log())).sum())
    assert _v(terms["consistency"]) > 0.0
    assert abs(_v(terms["consistency"]) - expected) < 1e-5


def test_t2_cr_response_only_mask():
    # token 0 differs wildly but its label is -100 -> excluded -> CR must be 0
    stu = torch.tensor([[[9.0, -9.0, 0.0], [0.0, 1.0, 2.0]]])
    ref = torch.tensor([[[-9.0, 9.0, 0.0], [0.0, 1.0, 2.0]]])
    batch = _batch(L=2, labels=torch.tensor([[-100, 5]]))

    def base(b):
        return SimpleNamespace(logits=ref.detach().expand(b["input_ids"].shape[0], -1, -1))

    terms = br.loss_terms(_cfg(balora_lambda1=1.0), FixedLogits(stu), _ctx(), batch, base)
    assert abs(_v(terms["consistency"])) < 1e-6


# ---------------------------------------------------------------- T3: DR (entropy)

def test_t3_dr_uniform_minimal_onehot_maximal():
    V = 10
    uni = FixedLogits(torch.zeros(1, 1, V))
    hot = FixedLogits(torch.tensor([[[10.0] + [-10.0] * (V - 1)]]))
    batch = _batch(L=1)
    cfg = _cfg(balora_lambda2=1.0, balora_top_k_entropy=0)  # full vocab (paper, D-002.2)
    t_uni = br.loss_terms(cfg, uni, _ctx(), batch, _no_base)["diversity"]
    t_hot = br.loss_terms(cfg, hot, _ctx(), batch, _no_base)["diversity"]
    assert t_uni < t_hot                                   # DR = -H: uniform is the minimum
    assert abs(_v(t_uni) - (-math.log(V))) < 1e-4          # -ln 10
    assert abs(_v(t_hot)) < 1e-4                           # one-hot -> H = 0 -> DR = 0
    # top-k proxy (U5): uniform logits restricted to top-4 -> -ln 4
    cfgk = _cfg(balora_lambda2=1.0, balora_top_k_entropy=4)
    t_k = br.loss_terms(cfgk, uni, _ctx(), batch, _no_base)["diversity"]
    assert abs(_v(t_k) - (-math.log(4))) < 1e-4


# ---------------------------------------------------------------- T4: SVDR (SVD ratio)

def _svdr_model(Z):
    return FixedLogits(Z)


def test_t4_svdr_exact_rank1_minus1_flat_minus_k_over_V():
    # rank-1 matrix: only one non-zero singular value -> ratio = 1 -> loss = -1
    Z1 = torch.outer(torch.randn(4), torch.randn(6))
    t = br.loss_terms(_cfg(balora_variant="nlu", balora_lambda3=1.0),
                      _svdr_model(Z1), _ctx(), _batch(L=4), _no_base)["svdr"]
    assert abs(_v(t) + 1.0) < 1e-5
    # flat spectrum: 10x8 with 8 equal singular values (orthonormal columns * 3)
    Q, _ = torch.linalg.qr(torch.randn(10, 8))
    Z2 = 3.0 * Q
    t = br.loss_terms(_cfg(balora_variant="nlu", balora_lambda3=1.0),
                      _svdr_model(Z2), _ctx(), _batch(L=10), _no_base)["svdr"]
    assert abs(_v(t) - (-5.0 / 8.0)) < 1e-5                # k=5 (NLU), V=8


def test_t4_svdr_randomized_rank1_close_to_minus1():
    Z1 = torch.outer(torch.randn(10), torch.randn(8))
    t = br.loss_terms(_cfg(balora_variant="nlg", balora_lambda3=1.0),
                      _svdr_model(Z1), _ctx(), _batch(L=10), _no_base)["svdr"]
    assert -1.0001 < _v(t) < -0.99


# ---------------------------------------------------------------- T5: schedule spec42

def test_t5_schedule_matches_spec42_by_hand():
    cfg = _cfg(balora_schedule="spec42", balora_warmup_ratio=0.2, balora_ramp_up_ratio=0.05)
    ctx = _ctx(total_steps=100)
    w, q = 0.2, 0.05

    def cos1(u):  # lambda1 factor, u > w
        return 0.5 * (1 + math.cos(math.pi * (u - w) / (1 - w)))

    cases = [  # (step, expected f1, expected f23)
        (0, 0.0, 0.0),
        (10, 0.5, 0.5),                       # linear warmup midpoint
        (20, 1.0, 1.0),                       # u = w exactly
        (50, cos1(0.5), 1.0),                 # cosine decays, focus holds
        (95, cos1(0.95), 1.0),                # u = 1-q: focus still at full value
        (98, cos1(0.98), (1 - 0.98) / q),     # ramp-down
        (100, 0.0, 0.0),
    ]
    for step, e1, e23 in cases:
        ctx.step = step
        f1, f2, f3 = br._schedule_factors(cfg, ctx)
        assert abs(f1 - e1) < 1e-9, (step, f1, e1)
        assert abs(f2 - e23) < 1e-9 and abs(f3 - e23) < 1e-9, (step, f2, e23)
    # schedule off -> raw lambdas
    assert br._schedule_factors(_cfg(), _ctx(step=50, total_steps=100)) == (1.0, 1.0, 1.0)
    # e2e: at step 0 every scheduled lambda is 0 -> reg skipped
    model = FixedLogits(torch.randn(1, 2, 8))
    terms = br.loss_terms(_cfg(balora_schedule="spec42", balora_lambda2=1.0), model,
                          _ctx(step=0, total_steps=100), _batch(L=2), _no_base)
    assert terms == {}


# ---------------------------------------------------------------- T6: Bp snapshot stability

def test_t6_bp_snapshot_not_drifting_after_adapter_update():
    model = MiniPissa()
    batch = _batch(B=2, L=5, V=8)
    ctx = _ctx(components=["Bp"], init_mode="pissa")
    cfg = _cfg(components=["Bp"], balora_lambda1=1.0, balora_distill_temp=1.0)
    captured = []

    def hook(m, inp, out):
        if not torch.is_grad_enabled():
            captured.append(out.logits.detach().clone())

    handle = model.register_forward_hook(hook)

    def must_not_call(*a, **k):
        raise AssertionError("Bp reference must use the snapshot swap, not base_forward")

    br.loss_terms(cfg, model, ctx, batch, must_not_call)          # call 1
    assert "_pissa_snapshot" in ctx.hooks
    snap = ctx.hooks["_pissa_snapshot"]
    assert set(snap) == {"lora_A", "lora_B"}
    assert torch.equal(snap["lora_A"], model.lora_A.detach())     # params restored after call 1
    # target of call 1 == exact W0 logits (snapshot-swap math equivalence)
    h = model.embed(batch["input_ids"])
    w0_logits = h @ model.W0.T
    assert torch.allclose(captured[0], w0_logits, atol=1e-5)

    with torch.no_grad():                                         # simulate a training update
        model.lora_A.add_(0.37)
        model.lora_B.add_(-0.21)
    updated_A = model.lora_A.detach().clone()

    br.loss_terms(cfg, model, ctx, batch, must_not_call)          # call 2
    handle.remove()
    assert torch.allclose(captured[1], captured[0], atol=1e-6)    # target did NOT drift
    assert torch.equal(model.lora_A.detach(), updated_A)          # current params restored bitwise


def test_t7_backward_grads_flow_all_regs():
    model = MiniPissa()
    batch = _batch(B=2, L=5, V=8)
    cfg = _cfg(components=["Bp"], balora_lambda1=1e-3, balora_lambda2=1e-1,
               balora_lambda3=1e-1, balora_distill_temp=2.0)
    ctx = _ctx(components=["Bp"], init_mode="pissa")
    terms = br.loss_terms(cfg, model, ctx, batch, None)
    assert set(terms) == {"consistency", "diversity", "svdr"}
    total = sum(terms.values())
    total.backward()
    assert model.lora_A.grad is not None and model.lora_B.grad is not None
    for g in (model.lora_A.grad, model.lora_B.grad):
        assert torch.isfinite(g).all()


def test_t8_reference_forward_skipped_when_lambda1_zero():
    model = FixedLogits(torch.randn(1, 2, 8))
    terms = br.loss_terms(_cfg(balora_lambda2=1.0, balora_lambda3=1.0), model,
                          _ctx(), _batch(L=2), _no_base)
    assert set(terms) == {"diversity", "svdr"}


def test_t10_provided_logits_match_internal_forward():
    """Pipeline passes its task-loss logits (Bẫy 11 OOM fix): same reg values and grads,
    and no second student forward happens when logits are provided."""
    ref_logits = torch.randn(2, 5, 8)

    def run(provide: bool):
        model = MiniPissa()
        batch = _batch(B=2, L=5, V=8)
        cfg = _cfg(components=["Br"], balora_lambda1=1e-2, balora_lambda2=1e-1,
                   balora_lambda3=1e-1, balora_distill_temp=2.0)
        ctx = _ctx(components=["Br"])
        base = lambda b: SimpleNamespace(logits=ref_logits)  # deterministic W0 reference
        kw = {}
        if provide:
            kw["logits_ft"] = model(**batch).logits
            model.forward = lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("loss_terms ran a second student forward"))
        terms = br.loss_terms(cfg, model, ctx, batch, base, **kw)
        sum(terms.values()).backward()
        return {k: _v(v) for k, v in terms.items()}, model.lora_A.grad.detach().clone()

    a, ga = run(False)
    b, gb = run(True)
    assert a.keys() == b.keys() == {"consistency", "diversity", "svdr"}
    for k in a:
        assert math.isclose(a[k], b[k], rel_tol=1e-4, abs_tol=1e-7), (k, a[k], b[k])
    assert torch.allclose(ga, gb, atol=1e-6)


# ---------------------------------------------------------------- NLU variant sanity

def test_t9_nlu_mse_covariance_exact_svd():
    B, C = 4, 3
    F_f = torch.randn(B, C)
    F_p = torch.randn(B, C)
    model = FixedLogits2D(F_f)

    def base(b):
        return SimpleNamespace(logits=F_p.detach())

    ctx = _ctx()
    terms = br.loss_terms(_cfg(balora_variant="nlu", balora_lambda1=1.0,
                               balora_lambda3=1.0), model, ctx, _batch(B=B), base)
    # eq 4: || F_p/||F_p|| - F_f/||F_f|| ||^2 (tensor-wise norm, U9)
    n_p, n_f = F_p / F_p.norm(), F_f / F_f.norm()
    assert abs(_v(terms["consistency"]) - float((n_p - n_f).pow(2).sum())) < 1e-5
    # eq 7 exact SVD with k=5: rank-1 -> -1
    Z1 = torch.outer(torch.randn(B), torch.randn(C))
    t = br.loss_terms(_cfg(balora_variant="nlu", balora_lambda3=1.0),
                      FixedLogits2D(Z1), ctx, _batch(B=B), base)["svdr"]
    assert abs(_v(t) + 1.0) < 1e-5
    # eq 5-6 covariance loss, hand-derived
    zc = F_f - F_f.mean(0, keepdim=True)
    Cmat = zc.T @ zc / (B - 1)
    off = Cmat - torch.diag(torch.diagonal(Cmat))
    t = br.loss_terms(_cfg(balora_variant="nlu", balora_lambda2=1.0),
                      model, ctx, _batch(B=B), base)["diversity"]
    assert abs(_v(t) - float(off.pow(2).sum() / C)) < 1e-5
