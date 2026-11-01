"""Unit tests for the AltLoRA optimizer (component `A`) — spec/altlora.md section 9.

CPU only, small tensors, no version asserts. Synthetic loss: least squares
L(A, B) = 1/2 * ||W0 + s*BA - Y||_F^2 with gradients from autograd (no hard-coded Eq. 1),
so the gradient contract (PEFT folds s into the forward) is part of what is tested.

Run: pytest tests/unit/test_altlora.py   (fallback: python tests/unit/run_altlora_checks.py)
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lorastack.adapters.base import AdapterContext  # noqa: E402
from lorastack.optim.altlora import AltLoRAOptimizer, _precond  # noqa: E402
from lorastack.registry import get_hooks  # noqa: E402


class _FakeLoraLayer(nn.Module):
    """PEFT lora.Linear param layout: lora_A/lora_B ModuleDicts of nn.Linear + scaling dict."""

    def __init__(self, k, d, r, seed, scaling=1.0, init_a=None, init_b=None):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, k, bias=False)})
        with torch.no_grad():
            if init_a is None:
                self.lora_A["default"].weight.normal_(
                    0, 0.5, generator=torch.Generator().manual_seed(seed))
            else:
                self.lora_A["default"].weight.copy_(init_a)
            if init_b is None:
                self.lora_B["default"].weight.zero_()  # paper init: B = 0
            else:
                self.lora_B["default"].weight.copy_(init_b)
        self.scaling = {"default": scaling}


class _FakeGraloraLayer(nn.Module):
    """PEFT gralora layer layout: ONE 3D param per factor, slice n = [in_blk, r] / [r, out_blk]
    stored transposed vs the paper (docs/peft_inventory.md section 3)."""

    def __init__(self, k_blk=2, in_blk=12, out_blk=10, r=4, seed=8):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        a = torch.empty(k_blk, in_blk, r)
        b = torch.zeros(k_blk, r, out_blk)
        for n in range(k_blk):
            a[n] = torch.randn(in_blk, r, generator=g) * 0.5
        self.gralora_A = nn.ParameterDict({"default": nn.Parameter(a)})
        self.gralora_B = nn.ParameterDict({"default": nn.Parameter(b)})
        self.scaling = {"default": 1.0}


def _cfg(lr=1e-2, momentum=0.0, wd=0.0, ridge=1e-2, order="B_first", variant="altlora",
         alpha=None, rank=4):
    return SimpleNamespace(lr=lr, altlora_momentum=momentum, weight_decay=wd,
                           altlora_ridge=ridge, altlora_order=order, altlora_variant=variant,
                           alpha=rank if alpha is None else alpha, rank=rank)


def _build(m, cfg):
    ctx = AdapterContext(exp_id="t", components=["A"], rank=cfg.rank)
    opt = get_hooks("A").make_optimizer(cfg, m, ctx)  # exercises the registry path
    assert isinstance(opt, AltLoRAOptimizer)
    return opt, ctx


def _ls_problem(k, d, seed, noise=0.1):
    g = torch.Generator().manual_seed(seed + 777)
    W0 = torch.randn(k, d, generator=g)
    Y = W0 + noise * torch.randn(k, d, generator=g)
    return W0, Y


def _loss(A, B, W0, Y, s=1.0):
    return 0.5 * ((W0 + s * (B @ A) - Y) ** 2).sum()


def _run_steps(opt, A, B, W0, Y, s, steps, record=None):
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = _loss(A, B, W0, Y, s)
        loss.backward()
        opt.step()
        if record is not None:
            record.append(loss.item())


def test_t1_monotone_loss_beta0():
    """T1: least squares, beta1=0, small eta -> loss decreases monotonically (+ numerics)."""
    k, d, r = 16, 24, 4
    m = _FakeLoraLayer(k, d, r, seed=1)
    opt, ctx = _build(m, _cfg(lr=1e-2, momentum=0.0, ridge=1e-2))
    assert len(ctx.ab_pairs) == 1 and ctx.ab_pairs[0][2] == ""  # (a, b, module_name)
    A, B = m.lora_A["default"].weight, m.lora_B["default"].weight
    W0, Y = _ls_problem(k, d, seed=1)
    losses = []
    _run_steps(opt, A, B, W0, Y, 1.0, 50, losses)
    assert all(losses[i + 1] <= losses[i] + 1e-6 for i in range(len(losses) - 1)), losses[:10]
    assert losses[-1] < losses[0]


def test_t2_precond_symmetric_psd_and_fallback():
    """T2: _precond returns a symmetric PSD inverse after the ridge; Cholesky ladder engages."""
    g = torch.Generator().manual_seed(4)
    Bm = torch.randn(16, 4, generator=g)
    gram = Bm.t() @ Bm
    P = _precond(gram, 1e-6)
    M = gram + 1e-6 * torch.eye(4)
    assert (P - P.t()).abs().max() / P.abs().max() < 1e-6  # symmetric
    assert (P @ M - torch.eye(4)).abs().max() < 1e-3       # really the inverse
    torch.linalg.cholesky(M)                               # PD after ridge -> cholesky OK
    assert (torch.linalg.eigvalsh(P) > 0).all()            # PSD (eigvalsh allowed in TESTS only)
    # indefinite gram (min eig -9e-4), eps=1e-6 -> the x10 ladder must recover at lam=1e-3
    q, _ = torch.linalg.qr(torch.randn(4, 4, generator=g))
    bad = q @ torch.diag(torch.tensor([1.0, 0.8, 0.5, -9e-4])) @ q.t()
    P2 = _precond(bad, 1e-6)
    assert torch.isfinite(P2).all()
    assert (P2 @ (bad + 1e-3 * torch.eye(4)) - torch.eye(4)).abs().max() < 1e-3
    # hopeless (min eig -1.0): even lam=1e-3 stays indefinite -> must raise, not return junk
    try:
        _precond(q @ torch.diag(torch.tensor([1.0, 0.8, 0.5, -1.0])) @ q.t(), 1e-6)
        raise AssertionError("expected RuntimeError from _precond")
    except RuntimeError:
        pass


def test_t3_transformation_invariance():
    """T3: (A, B) -> (QA, BQ^-1) with invertible Q gives the same BA trajectory (tol O(lambda))."""
    torch.manual_seed(5)
    k, d, r = 16, 24, 4
    W0, Y = _ls_problem(k, d, seed=5)
    g = torch.Generator().manual_seed(6)
    A1 = torch.randn(r, d, generator=g) * 0.5
    B1 = torch.randn(k, r, generator=g) * 0.5
    Q = torch.eye(r) + 0.1 * torch.randn(r, r, generator=g)
    assert abs(Q.det()) > 0.3 and torch.linalg.cond(Q) < 20  # well-conditioned Q
    Qinv = torch.inverse(Q)
    inits = [(A1.clone(), B1.clone()), (Q @ A1, B1 @ Qinv)]
    opts, params = [], []
    for i, (Ai, Bi) in enumerate(inits):
        m = _FakeLoraLayer(k, d, r, seed=7 + i, init_a=Ai, init_b=Bi)
        opt, _ = _build(m, _cfg(lr=1e-2, momentum=0.9, ridge=1e-6))
        opts.append(opt)
        params.append((m.lora_A["default"].weight, m.lora_B["default"].weight))
    prods0 = [(B @ A).detach().clone() for A, B in params]
    for _ in range(10):  # 5 B-steps + 5 A-steps, both phases
        for opt, (A, B) in zip(opts, params):
            _run_steps(opt, A, B, W0, Y, 1.0, 1)
    d1 = (params[0][1] @ params[0][0]).detach() - prods0[0]
    d2 = (params[1][1] @ params[1][0]).detach() - prods0[1]
    assert d1.norm() > 1e-6  # the trajectory actually moved
    assert (d1 - d2).norm() / d1.norm() < 1e-2


def test_t4_b_first_step0_and_b0_noop():
    """T4: B-first order; with B = 0 the A-step is an exact no-op (g_A = s*B^T*R = 0)."""
    k, d, r = 16, 24, 4
    m = _FakeLoraLayer(k, d, r, seed=2)
    opt, _ = _build(m, _cfg(lr=1e-2, ridge=1e-2))
    A, B = m.lora_A["default"].weight, m.lora_B["default"].weight
    W0, Y = _ls_problem(k, d, seed=2)
    A0 = A.detach().clone()
    # t=0 B-phase with zero B-grad: B stays exactly 0, A untouched
    B.grad, A.grad = torch.zeros_like(B), torch.zeros_like(A)
    opt.step()
    assert B.detach().abs().sum() == 0 and torch.equal(A.detach(), A0)
    # t=1 A-phase via autograd: with B = 0 the gradient itself is zero -> A bit-exact unchanged
    opt.zero_grad(set_to_none=True)
    _loss(A, B, W0, Y, 1.0).backward()
    opt.step()
    assert torch.equal(A.detach(), A0)
    # t=2 B-phase: B moves off zero; t=3 A-phase: now A moves too
    opt.zero_grad(set_to_none=True)
    _loss(A, B, W0, Y, 1.0).backward()
    opt.step()
    assert B.detach().abs().sum() > 0
    opt.zero_grad(set_to_none=True)
    _loss(A, B, W0, Y, 1.0).backward()
    opt.step()
    assert not torch.equal(A.detach(), A0)
    # parity survives a state_dict roundtrip: resume lands on the A-phase, not a fresh B-step
    m2 = _FakeLoraLayer(k, d, r, seed=12)
    opt_a, _ = _build(m2, _cfg(lr=1e-2, ridge=1e-2))
    A2, B2 = m2.lora_A["default"].weight, m2.lora_B["default"].weight
    _run_steps(opt_a, A2, B2, W0, Y, 1.0, 1)  # t=0 B-phase done (_steps = 1)
    B_after0, A0_2 = B2.detach().clone(), A2.detach().clone()
    opt_b, _ = _build(m2, _cfg(lr=1e-2, ridge=1e-2))
    opt_b.load_state_dict(opt_a.state_dict())
    _run_steps(opt_b, A2, B2, W0, Y, 1.0, 1)
    assert torch.equal(B2.detach(), B_after0)      # NOT another B-step
    assert not torch.equal(A2.detach(), A0_2)      # it was the A-phase


def test_t5_state_budget():
    """T5: per pair exactly 2 momentum tensors; total state O(kr+dr), nothing of size k*d."""
    k, d, r = 16, 24, 4
    m = _FakeLoraLayer(k, d, r, seed=3, init_b=torch.randn(k, r, generator=torch.Generator().manual_seed(3)) * 0.5)
    opt, _ = _build(m, _cfg(lr=1e-2, momentum=0.9, ridge=1e-2))
    A, B = m.lora_A["default"].weight, m.lora_B["default"].weight
    W0, Y = _ls_problem(k, d, seed=3)
    _run_steps(opt, A, B, W0, Y, 1.0, 2)  # one B-step + one A-step
    st = opt.state[A]
    u = st["units"][0]
    assert st["steps"] == 2
    assert set(u) == {"m_a", "m_b", "p_a_inv", "x_ba", "p_b_inv", "x_ab", "n_a", "n_b"}
    assert u["n_b"] == 1 and u["n_a"] == 1
    assert u["m_a"].shape == (r, d) and u["m_b"].shape == (k, r)
    assert u["m_a"].dtype == torch.float32 and u["m_b"].dtype == torch.float32
    tensors = [t for t in u.values() if isinstance(t, torch.Tensor)]
    assert all(t.numel() < k * d for t in tensors)               # no full-size buffer
    assert sum(t.numel() for t in tensors) <= 3 * (k * r + d * r)  # O(kr+dr) + small r^2 caches


def test_t6_gralora_slice_independence():
    """T6 (X4): the 3D GraLoRA params yield one state entry per slice; a slice with zero grad
    is untouched (params and momentum) while the other slice updates."""
    m = _FakeGraloraLayer(k_blk=2, in_blk=12, out_blk=10, r=4, seed=8)
    opt, ctx = _build(m, _cfg(lr=1e-2, momentum=0.9, ridge=1e-2))
    a3, b3 = m.gralora_A["default"], m.gralora_B["default"]
    assert len(ctx.ab_pairs) == 1 and ctx.ab_pairs[0][0] is a3 and ctx.ab_pairs[0][1] is b3
    units = opt.state[a3]["units"]
    assert len(units) == 2
    g = torch.Generator().manual_seed(9)
    a3.grad, b3.grad = torch.zeros_like(a3), torch.zeros_like(b3)
    b3.grad[0].normal_(0, 1.0, generator=g)  # B-phase: grad on slice 0 only
    snap_a0, snap_a1 = a3.detach()[0].clone(), a3.detach()[1].clone()
    snap_b0, snap_b1 = b3.detach()[0].clone(), b3.detach()[1].clone()
    opt.step()
    assert b3.detach()[1].abs().sum() == 0 and torch.equal(b3.detach()[1], snap_b1)
    assert not torch.equal(b3.detach()[0], snap_b0)  # b init is 0 -> moved off zero
    a3.grad, b3.grad = torch.zeros_like(a3), torch.zeros_like(b3)
    a3.grad[0].normal_(0, 1.0, generator=g)  # A-phase: grad on slice 0 only
    opt.step()
    assert torch.equal(a3.detach()[1], snap_a1)      # slice 1 untouched (zero grad)
    assert not torch.equal(a3.detach()[0], snap_a0)  # slice 0 moved
    # independent state: both slices stepped, but slice 1 stayed all-zero (zero grads)
    assert units[0]["n_b"] == 1 and units[0]["n_a"] == 1
    assert units[1]["n_b"] == 1 and units[1]["n_a"] == 1
    assert units[0]["m_b"].abs().sum() > 0 and units[0]["m_a"].abs().sum() > 0
    assert units[1]["m_b"].abs().sum() == 0 and units[1]["m_a"].abs().sum() == 0
    assert units[0]["m_a"].shape == (4, 12) and units[0]["m_b"].shape == (10, 4)  # paper shapes


def test_t7_fp16_params():
    """fp16/T4: fp32 islands — state stays fp32, params keep fp16, large grads do not NaN."""
    k, d, r = 16, 24, 4
    m = _FakeLoraLayer(k, d, r, seed=10).to(torch.float16)
    opt, _ = _build(m, _cfg(lr=1e-3, momentum=0.9, ridge=1e-2))
    A, B = m.lora_A["default"].weight, m.lora_B["default"].weight
    W0, Y = _ls_problem(k, d, seed=10)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        (100.0 * _loss(A, B, W0, Y, 1.0)).backward()  # big grads, like an unscaled fp16 step
        opt.step()
    assert A.dtype == torch.float16 and B.dtype == torch.float16
    assert torch.isfinite(A.detach()).all() and torch.isfinite(B.detach()).all()
    st = opt.state[A]
    u = st["units"][0]
    assert u["m_a"].dtype == torch.float32 and u["m_b"].dtype == torch.float32
    assert u["p_b_inv"].dtype == torch.float32
    assert B.detach().abs().sum() > 0  # updates actually landed despite the fp16 write-back


def test_t8_scaler_skip_keeps_phase():
    """A GradScaler-skipped step (inf grad) must NOT advance the phase parity."""
    k, d, r = 16, 24, 4
    m = _FakeLoraLayer(k, d, r, seed=11)
    opt, _ = _build(m, _cfg(lr=1e-2, momentum=0.9, ridge=1e-2))
    A, B = m.lora_A["default"].weight, m.lora_B["default"].weight
    W0, Y = _ls_problem(k, d, seed=11)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    opt.zero_grad(set_to_none=True)
    scaler.scale(_loss(A, B, W0, Y, 1.0)).backward()
    scaler.unscale_(opt)
    scaler.step(opt)   # real step 0 -> B-phase
    scaler.update()
    A0 = A.detach().clone()
    B_after0 = B.detach().clone()
    assert B_after0.abs().sum() > 0 and torch.equal(A.detach(), A0)
    opt.zero_grad(set_to_none=True)
    scaler.scale(_loss(A, B, W0, Y, 1.0)).backward()
    B.grad[0, 0] = float("inf")  # simulated overflow before the (internal) unscale
    scaler.step(opt)   # must detect inf and skip optimizer.step entirely
    scaler.update()
    assert torch.equal(B.detach(), B_after0)   # skipped: nothing moved
    opt.zero_grad(set_to_none=True)
    scaler.scale(_loss(A, B, W0, Y, 1.0)).backward()
    scaler.unscale_(opt)
    scaler.step(opt)   # next real step must be the A-phase
    scaler.update()
    assert torch.equal(B.detach(), B_after0)   # B not updated again
    assert not torch.equal(A.detach(), A0)     # A updated (parity intact)


def test_t9_scaling_correction_1_over_s2():
    """Theorem 1: the 1/s^2 factor cancels the s inside autograd gradients, so the s*BA
    (i.e. dW) trajectory is nearly independent of the module scaling (error O(lambda))."""
    k, d, r = 16, 24, 4
    W0, Y = _ls_problem(k, d, seed=12)
    g = torch.Generator().manual_seed(13)
    A0 = torch.randn(r, d, generator=g) * 0.5
    B0 = torch.randn(k, r, generator=g) * 0.5
    curves = []
    for s in (1.0, 2.0):
        m = _FakeLoraLayer(k, d, r, seed=14, scaling=s, init_a=A0, init_b=B0 / s)
        opt, _ = _build(m, _cfg(lr=1e-2, momentum=0.9, ridge=1e-6))
        A, B = m.lora_A["default"].weight, m.lora_B["default"].weight
        losses = []
        _run_steps(opt, A, B, W0, Y, s, 10, losses)  # loss already contains the s
        curves.append(losses)
    for l1, l2 in zip(curves[0], curves[1]):
        assert abs(l1 - l2) <= 1e-4 * max(1.0, abs(l1)), (curves[0][:3], curves[1][:3])
