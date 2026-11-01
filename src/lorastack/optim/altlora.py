"""AltLoRA optimizer (component `A`) — alternating projections on (A, B) adapter pairs.

Clean-room reimplementation from spec/altlora.md (paper: "AltLoRA: Towards Better Gradient
Approximation in Low-Rank Adaptation with Alternating Projections", NeurIPS 2025,
arXiv 2505.12455v2). Author code was not read.

One optimizer step updates exactly ONE factor per (A, B) pair; phases alternate with the
parity of an internal counter (B-first by default, paper Table 5: with the standard B=0 init
the first A-step is a no-op). The active factor's raw autograd gradient is preconditioned
toward the full gradient (Theorem 1 + Corollary 1: multiply by (Gram + lambda*I)^-1 and by
1/s^2 to undo the scaling s = alpha/r that PEFT folds into the forward), and the first-order
momentum is realigned between successive low-rank subspaces (Theorem 2 / Corollary 2).
State per pair stays O(kr + dr): two momentum buffers plus four r x r caches.

Numerics (fp16/T4): all preconditioner math runs in fp32 islands (Cholesky inverse with
ridge fallback lambda x10 up to 3 times, no eigvalsh/SVD); parameters are written back in
their original dtype. GradScaler works unchanged: the pipeline unscales before step(), and a
scaler-skipped step (inf/nan grads) never advances the phase parity because the counter only
moves inside step(). The (Gram + lambda*I)^-1 preconditioner assumes generically full-rank
factors (in_blk, out_blk >= r — true for all lab shapes, e.g. r=32 vs in_blk>=224); a slice
with r > in_blk is structurally rank-deficient and the ridge ladder raises loudly instead of
taking a ~1/lambda step along the null space.

GraLoRA (X4): gralora_A.default [k, in_blk, r] / gralora_B.default [k, r, out_blk] are ONE
3D parameter each (docs/peft_inventory.md section 3); slice n — stored TRANSPOSED vs the
paper layout — is one independently updated unit with its own state (no cross-slice
coupling).
"""
from __future__ import annotations

import torch

from lorastack.registry import register

_BETA2 = 0.999      # Algorithm 2 second-moment decay (paper section 5.1)
_EPS_PLUS = 1e-6    # Algorithm 2 denominator epsilon; paper does not state it (spec 8.4)
_RIDGE_RETRIES = 3  # lambda <- 10*lambda up to 3 times before raising (spec section 7)


def _precond(gram, eps):
    """Return (gram + lam*I)^-1 as an fp32 r x r tensor via Cholesky (no eigvalsh/SVD).

    `gram` is the raw Gram matrix (B^T B or A A^T, ridge NOT yet added). The matrix is
    symmetrized, ridged with `eps`; on Cholesky failure (numerically not PD) the ridge is
    multiplied by 10 up to _RIDGE_RETRIES times, then RuntimeError (spec section 7).
    """
    g = gram.detach().float()
    lam = float(eps)
    eye = torch.eye(g.shape[0], dtype=g.dtype, device=g.device)
    last = None
    for _ in range(_RIDGE_RETRIES + 1):
        m = g + lam * eye
        m = 0.5 * (m + m.t())
        try:
            return torch.cholesky_inverse(torch.linalg.cholesky(m))
        except Exception as exc:  # LAPACK info > 0: not positive definite
            last = exc
            lam *= 10.0
    raise RuntimeError(f"AltLoRA _precond: Cholesky failed for ridge up to {lam:.1e}") from last


class _Unit:
    """One independently updated (A, B) factorization, in PAPER orientation.

    2D PEFT LoRA: a=[r, d_in], b=[d_out, r] (native == paper orientation, one unit per pair).
    GraLoRA slice n: native a=[in_blk, r], b=[r, out_blk] — transposed vs paper, so the paper
    orientation is the .t() view. All math below uses paper shapes: a=[r, d], b=[out, r].
    """

    __slots__ = ("a_param", "b_param", "a_sl", "b_sl", "transposed", "inv_s2")

    def __init__(self, a_param, b_param, a_sl, b_sl, transposed, inv_s2):
        self.a_param = a_param
        self.b_param = b_param
        self.a_sl = a_sl
        self.b_sl = b_sl
        self.transposed = transposed
        self.inv_s2 = inv_s2  # 1/s^2 from Theorem 1 / Corollary 1

    def _view(self, t, sl):
        x = t if sl is None else t[sl]
        return x.t() if self.transposed else x

    def a_data(self):
        return self._view(self.a_param.data, self.a_sl).float()

    def b_data(self):
        return self._view(self.b_param.data, self.b_sl).float()

    def a_grad(self):
        if self.a_param.grad is None:
            return None
        return self._view(self.a_param.grad, self.a_sl).float()

    def b_grad(self):
        if self.b_param.grad is None:
            return None
        return self._view(self.b_param.grad, self.b_sl).float()

    def write_a(self, a_new):
        native = a_new.t() if self.transposed else a_new
        dst = self.a_param.data if self.a_sl is None else self.a_param.data[self.a_sl]
        dst.copy_(native.to(dst.dtype))  # keeps param dtype (fp16-safe)

    def write_b(self, b_new):
        native = b_new.t() if self.transposed else b_new
        dst = self.b_param.data if self.b_sl is None else self.b_param.data[self.b_sl]
        dst.copy_(native.to(dst.dtype))

    def m_a_shape(self):
        sh = list(self.a_param.shape)
        if self.a_sl is not None:
            sh = sh[1:]
        if self.transposed:
            sh.reverse()
        return tuple(sh)  # [r, d]

    def m_b_shape(self):
        sh = list(self.b_param.shape)
        if self.b_sl is not None:
            sh = sh[1:]
        if self.transposed:
            sh.reverse()
        return tuple(sh)  # [out, r]


class _Pair:
    """One ab_pairs entry: the (a_param, b_param) tensors of one adapted module."""

    __slots__ = ("a_param", "b_param", "module_name", "units")

    def __init__(self, a_param, b_param, module_name, units):
        self.a_param = a_param
        self.b_param = b_param
        self.module_name = module_name
        self.units = units


def _adapter_side(seg):
    if seg in ("lora_A", "lora_B") or seg.startswith("gralora_A") or seg.startswith("gralora_B"):
        return seg
    return None


def _module_scaling(mod, adapter, cfg):
    """s from the PEFT layer (`layer.scaling[adapter]`); fallback alpha/r (spec section 2)."""
    sc = getattr(mod, "scaling", None)
    if isinstance(sc, dict):
        if adapter in sc:
            return float(sc[adapter])
        if len(sc) == 1:
            return float(next(iter(sc.values())))
    # TODO(author-code): adapter modules without PEFT `.scaling` fall back to alpha/r;
    # revisit if such targets (e.g. raw nn.Parameters) are ever enabled.
    return cfg.alpha / cfg.rank


def _make_pair(cfg, a, b, mod, module_name, adapter):
    if a.dim() != b.dim():
        raise ValueError(f"AltLoRA: A/B param dim mismatch on {module_name}: {tuple(a.shape)} vs {tuple(b.shape)}")
    s = _module_scaling(mod, adapter, cfg)
    inv_s2 = 1.0 / (s * s)
    if a.dim() == 3:  # GraLoRA (X4): k per-slice units of the single 3D parameters
        k = a.shape[0]
        if b.shape[0] != k:
            raise ValueError(f"AltLoRA: GraLoRA block count mismatch on {module_name}: {tuple(a.shape)} vs {tuple(b.shape)}")
        units = [_Unit(a, b, n, n, True, inv_s2) for n in range(k)]
    elif a.dim() == 2:
        units = [_Unit(a, b, None, None, False, inv_s2)]
    else:
        raise ValueError(f"AltLoRA: unsupported param dim {a.dim()} on {module_name}")
    return _Pair(a, b, module_name, units)


def _collect_pairs(cfg, model):
    """Pair adapter A/B weight params by full parameter path (segment-based, robust to the
    PEFT module nesting: ...q_proj.lora_A.default.weight / ...q_proj.gralora_A.default)."""
    a_map, b_map = {}, {}
    for key, p in model.named_parameters():
        segs = key.split(".")
        idx = next((i for i, s in enumerate(segs) if _adapter_side(s)), -1)
        if idx < 0:
            continue
        side, prefix = segs[idx], ".".join(segs[:idx])
        rest = segs[idx + 2:]  # skip the adapter-name segment
        if side.startswith("lora_") and rest != ["weight"]:
            continue  # e.g. lora bias — not one of the (A, B) weight factors
        if side.startswith("gralora_") and rest:
            continue
        (a_map if side.endswith("_A") else b_map)[(prefix, segs[idx + 1] if idx + 1 < len(segs) else "")] = p
    modules = dict(model.named_modules())
    pairs = []
    for prefix, adapter in sorted(set(a_map) & set(b_map)):
        mod = modules.get(prefix, model)
        pairs.append(_make_pair(cfg, a_map[(prefix, adapter)], b_map[(prefix, adapter)], mod, prefix, adapter))
    covered = {id(p) for pair in pairs for p in (pair.a_param, pair.b_param)}
    stray = [k for k, p in model.named_parameters() if p.requires_grad and id(p) not in covered]
    if stray:  # fail loud instead of silently never training those params
        raise ValueError(f"AltLoRA: trainable adapter params outside (A, B) pairs: {stray[:4]}")
    return pairs


class AltLoRAOptimizer(torch.optim.Optimizer):
    """Alternating-projection optimizer over (A, B) adapter pairs.

    Algorithm 1 by default; Algorithm 2 with variant="altlora_plus" (second moment V^A/V^B,
    no bias correction, denominator sqrt(V)+eps, weight decay added after the division —
    state doubles but stays O(kr+dr), and transformation invariance is lost, paper 4.2).
    Phase parity lives in step() (survives GradScaler skips) and is checkpointed as the
    "steps" entry of the first parameter's state.
    """

    def __init__(self, pairs, params, lr, beta1=0.9, weight_decay=0.0, ridge=1e-6,
                 order="B_first", variant="altlora"):
        if order not in ("B_first", "A_first"):
            raise ValueError(f"altlora_order must be 'B_first'|'A_first', got {order!r}")
        if variant not in ("altlora", "altlora_plus"):
            raise ValueError(f"altlora_variant must be 'altlora'|'altlora_plus', got {variant!r}")
        if not params:
            raise ValueError("AltLoRA: no (A, B) adapter pairs found in model")
        super().__init__(params, {"lr": lr})
        self.pairs = pairs
        self.beta1 = float(beta1)
        self.wd = float(weight_decay)
        self.ridge = float(ridge)
        self.order = order
        self.plus = variant == "altlora_plus"
        self._steps = 0
        for pair in pairs:
            self.state[pair.a_param]["units"] = [self._new_unit_state(u) for u in pair.units]
        self.state[self.param_groups[0]["params"][0]]["steps"] = 0

    def _new_unit_state(self, unit):
        st = {
            "m_a": torch.zeros(unit.m_a_shape(), dtype=torch.float32, device=unit.a_param.device),
            "m_b": torch.zeros(unit.m_b_shape(), dtype=torch.float32, device=unit.b_param.device),
            "p_a_inv": None, "x_ba": None, "p_b_inv": None, "x_ab": None,
            "n_a": 0, "n_b": 0,
        }
        if self.plus:  # Algorithm 2: V^A, V^B second moments (doubles state, still O(kr+dr))
            st["v_a"] = torch.zeros(unit.m_a_shape(), dtype=torch.float32, device=unit.a_param.device)
            st["v_b"] = torch.zeros(unit.m_b_shape(), dtype=torch.float32, device=unit.b_param.device)
        return st

    # -- phase steps -----------------------------------------------------------

    def _b_step(self, unit, st, eta):
        g_b = unit.b_grad()
        if g_b is None:
            return
        if st["n_a"] > 0:
            p_a = st["p_a_inv"]  # (A A^T + lam I)^-1 cached at the last A-step (A unchanged since)
        else:
            a_cur = unit.a_data()  # first B-step: no cache yet (spec section 4)
            p_a = _precond(a_cur @ a_cur.t(), self.ridge)
        gt = unit.inv_s2 * (g_b @ p_a)  # [out, r]
        if st["n_b"] > 0:
            # realign momentum A_old -> A_new column space (Corollary 2): M @ (A_old A_new^T) @ P
            m = self.beta1 * (st["m_b"] @ st["x_ba"] @ p_a) + (1.0 - self.beta1) * gt
        else:
            m = (1.0 - self.beta1) * gt  # first use: M_0 = 0, no realignment (spec 8.3)
        b_old = unit.b_data()
        if self.plus:
            v = _BETA2 * st["v_b"] + (1.0 - _BETA2) * (gt * gt)
            st["v_b"] = v
            direction = m / (v.sqrt() + _EPS_PLUS) + self.wd * b_old
        else:
            direction = m + self.wd * b_old  # decoupled weight decay -eta*gamma*B (spec 8.8)
        unit.write_b(b_old - eta * direction)
        st["m_b"] = m
        b_new = unit.b_data()  # re-read: caches must reflect the actually stored (rounded) B
        st["p_b_inv"] = _precond(b_new.t() @ b_new, self.ridge)
        st["x_ab"] = b_new.t() @ b_old  # for the next A-step realignment
        st["n_b"] += 1

    def _a_step(self, unit, st, eta):
        g_a = unit.a_grad()
        if g_a is None:
            return
        if st["n_b"] > 0:
            p_b = st["p_b_inv"]  # (B^T B + lam I)^-1 cached at the last B-step (B unchanged since)
        else:
            b_cur = unit.b_data()
            p_b = _precond(b_cur.t() @ b_cur, self.ridge)
        gt = unit.inv_s2 * (p_b @ g_a)  # [r, d]
        if st["n_a"] > 0:
            # realign momentum B_old -> B_new: P @ (B_new^T B_old) @ M (ridge applied
            # symmetrically; Corollary 2 only states the B side, spec 8.2)
            m = self.beta1 * (p_b @ st["x_ab"] @ st["m_a"]) + (1.0 - self.beta1) * gt
        else:
            m = (1.0 - self.beta1) * gt
        a_old = unit.a_data()
        if self.plus:
            v = _BETA2 * st["v_a"] + (1.0 - _BETA2) * (gt * gt)
            st["v_a"] = v
            direction = m / (v.sqrt() + _EPS_PLUS) + self.wd * a_old
        else:
            direction = m + self.wd * a_old
        unit.write_a(a_old - eta * direction)
        st["m_a"] = m
        a_new = unit.a_data()
        st["p_a_inv"] = _precond(a_new @ a_new.t(), self.ridge)
        st["x_ba"] = a_old @ a_new.t()  # for the next B-step realignment
        st["n_a"] += 1

    # -- torch.optim.Optimizer API --------------------------------------------

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        eta = self.param_groups[0]["lr"]
        phase_b_now = (self._steps % 2 == 0) == (self.order == "B_first")
        phase = "B" if phase_b_now else "A"
        for pair in self.pairs:
            units = self.state[pair.a_param]["units"]
            for unit, st in zip(pair.units, units):
                (self._b_step if phase == "B" else self._a_step)(unit, st, eta)
        self._steps += 1
        self.state[self.param_groups[0]["params"][0]]["steps"] = self._steps
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        first = self.param_groups[0]["params"][0]
        self._steps = int(self.state.get(first, {}).get("steps", 0))


def make_optimizer(cfg, model, ctx):
    """Registry hook `A` (see src/lorastack/registry.py): build the AltLoRA optimizer.

    Collects the (lora_A, lora_B) weight pairs per adapted module itself and fills
    ctx.ab_pairs with (a_param, b_param, module_name) triples (the pipeline only falls back
    to its own collection when this hook leaves ctx.ab_pairs empty). GraLoRA pairs stay one
    entry per module here and are expanded per-slice inside the optimizer (X4). All factors
    share one lr in a single param group (Theorem 3), so lr schedulers editing
    param_groups[0]["lr"] keep working.
    """
    pairs = _collect_pairs(cfg, model)
    if ctx is not None:
        ctx.ab_pairs = [(p.a_param, p.b_param, p.module_name) for p in pairs]
    params, seen = [], set()
    for pair in pairs:
        for p in (pair.a_param, pair.b_param):
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)
    return AltLoRAOptimizer(
        pairs, params,
        lr=cfg.lr,
        beta1=cfg.altlora_momentum,
        weight_decay=cfg.weight_decay,
        ridge=cfg.altlora_ridge,
        order=getattr(cfg, "altlora_order", "B_first"),
        variant=getattr(cfg, "altlora_variant", "altlora"),
    )


register("A", make_optimizer=make_optimizer)
