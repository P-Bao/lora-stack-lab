"""V2 black-box: GoRA author code (D:\\...\\GoRA) vs lorastack reimpl — NO author file is
modified, nothing is installed. Missing third-party imports are satisfied with throwaway
in-memory stub objects (import-time glue only).

Black-box protocol — author surfaces used (all disclosed):
  - grep of def/class lines in gora.py + narrow ranges 24-58, 298-315, 323-330, 370-383
    (signatures and CALL SITES only, to learn argument conventions);
  - top-level import lines of 3 files in the author `common` package;
  - CLI flags of example_llama3.1_metamathqa.sh and defaults of gora_t5_helpers.py
    (config surface).
No algorithm bodies were studied. All probes below feed FIXED synthetic tensors
(seed 0, W 64x32, G rank-4) and compare I/O against lorastack reimpl.

Run:  python tests/blackbox/run_gora_blackbox.py
"""
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

GORACODE = Path(r"D:\Documents\NCKH\NLP\ReaserchTeamNLP\LoRA\GoRA\core code gora")
MYTRANSFORMERS = Path(r"D:\Documents\NCKH\NLP\ReaserchTeamNLP\LoRA\GoRA\MyTransformers")

SEED = 0
M, N = 64, 32          # W: (out, in)
RANK_G = 4
R_REF = 8
R_MAX = 32
R_MIN = 4              # author CLI default --gora-min-rank 4 (matches our paper B.3 r_ref/2)

report = []


def log(tag, status, detail=""):
    report.append((tag, status, detail))
    print(f"[{status:9s}] {tag:52s} {str(detail)[:170]}")


class _MagicAny:
    """Throwaway stand-in for a missing third-party module: callable, iterable,
    context-manager, attribute-producing. Import-time glue only — any REAL use in math
    will produce garbage, so probes that touch a stub must be treated as invalid."""
    def __init__(self, name="stub"):
        import importlib.machinery
        object.__setattr__(self, "__name__", name)
        # real ModuleSpec so importlib.util.find_spec(stub) does not blow up third-party
        # availability probes (transformers asks find_spec("deepspeed") at import time)
        object.__setattr__(self, "__spec__", importlib.machinery.ModuleSpec(name, None))

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        v = _MagicAny(f"{getattr(self, '__name__', 'stub')}.{attr}")
        object.__setattr__(self, attr, v)
        return v

    def __call__(self, *a, **k): return _MagicAny()
    def __getitem__(self, k): return _MagicAny()
    def __iter__(self): return iter(())
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __repr__(self): return "<MagicAny>"
    def __str__(self): return "<MagicAny>"
    def __bool__(self): return True


_MAGIC_ROOTS = set()


def stub_import(fn, tries=60):
    """Run fn(); on ModuleNotFoundError stub the missing non-`common` root and retry."""
    for _ in range(tries):
        try:
            return fn()
        except ModuleNotFoundError as e:
            name = e.name or "?"
            root = name.split(".")[0]
            if name.startswith("common") or root in _MAGIC_ROOTS:
                raise
            _MAGIC_ROOTS.add(root)
            sys.modules[root] = _MagicAny(root)
    raise RuntimeError("stub loop exhausted")


def _construct_probe(cls, seed_kwargs):
    """Black-box constructor: fill missing required args from TypeError messages."""
    try:
        return cls()
    except Exception:
        pass
    kwargs = dict(seed_kwargs)
    for _ in range(30):
        try:
            return cls(**kwargs)
        except TypeError as e:
            m = re.search(r"missing \d+ required positional arguments?: '(\w+)'", str(e))
            if not m:
                raise
            kwargs[m.group(1)] = 8
    return None


# ---------------------------------------------------------------- synthetic tensors
torch.manual_seed(SEED)
W = torch.randn(M, N, dtype=torch.float32)
g = torch.Generator().manual_seed(SEED)
U = torch.randn(M, RANK_G, generator=g)
V = torch.randn(RANK_G, N, generator=g)
G = U @ V  # rank-4 gradient, fixed

ours_I = float((W * G).abs().mean())  # our Eq.(5)

# ---------------------------------------------------------------- A. import
gora = None
try:
    common = types.ModuleType("common")
    common.__path__ = [str(MYTRANSFORMERS / "common")]
    sys.modules["common"] = common
    sys.path.insert(0, str(GORACODE))
    gora = stub_import(lambda: __import__("gora"))
    log("A import gora", "OK", str(getattr(gora, "__file__", GORACODE / "gora.py")))
except Exception as e:
    log("A import gora", "DEFERRED", f"{type(e).__name__}: {e}")

author_lora = None
try:
    author_lora = stub_import(lambda: __import__("common.lora_modules.lora", fromlist=["lora"]))
    log("A' import common.lora_modules.lora", "OK", "")
except Exception as e:
    log("A' import common.lora_modules.lora", "DEFERRED", f"{type(e).__name__}: {str(e)[:120]}")

if gora is not None and author_lora is not None:
    # ---------------------------------------------------------------- E0. author layer
    layer = None
    try:
        import dataclasses
        if dataclasses.is_dataclass(gora.LoRAConfig):
            fields = {f.name: (f.default if f.default is not dataclasses.MISSING else
                               ("<required>" if f.default_factory is dataclasses.MISSING else "<factory>"))
                      for f in dataclasses.fields(gora.LoRAConfig)}
            log("E0 LoRAConfig fields", "OK", str(fields))
        author_cfg = _construct_probe(gora.LoRAConfig, dict(
            lora_rank=R_REF, lora_scaler=1.0, in_features=N, out_features=M,
        ))
        layer = gora.LinearWithGoRA(author_cfg)
        attrs = [a for a in dir(layer) if not a.startswith("_")]
        log("E0 LinearWithGoRA", "OK", f"constructed; public attrs={attrs[:26]}")
        wp = getattr(layer, "weight", None)
        if wp is None and hasattr(layer, "base_layer"):
            wp = layer.base_layer.weight
        if wp is None:
            raise RuntimeError("no base weight attribute found")
        with torch.no_grad():
            wp.copy_(W)
    except Exception as e:
        log("E0 LinearWithGoRA", "DEFERRED", f"{type(e).__name__}: {str(e)[:200]}")

    if layer is not None:
        # ---------------------------------------------------------- B. compute_importance
        wp = layer.weight if hasattr(layer, "weight") else layer.base_layer.weight
        wp.grad = G.clone()
        wp.grad_stored = G.clone()  # author convention: grad stored as tensor attribute
        features_int = N + M        # call site: features = module.in_features + out_features
        got, ok_cfg, err_last = None, None, ""
        for t in ("union_mean", "union_max", "mean", "max", "sum", "product",
                  "union_mean_scaled", "grad_mean", "abs_mean"):
            for sc in (False, 1.0):
                try:
                    got = gora.compute_importance(wp.data, wp.grad_stored, features_int, sc, t, R_REF, R_MAX)
                    ok_cfg = (t, sc)
                    break
                except Exception as e:
                    err_last = f"{t}/{sc!r}: {type(e).__name__}: {str(e)[:80]}"
            if got is not None:
                break
        if got is None:
            log("B compute_importance", "DEFERRED", f"no variant worked; last error {err_last}")
        else:
            def _flat(x, depth=0):
                if torch.is_tensor(x):
                    return f"tensor{tuple(x.shape)}"
                if isinstance(x, tuple):
                    return "(" + ", ".join(_flat(y, depth + 1) for y in x) + ")"
                return f"{type(x).__name__}:{x if isinstance(x, (int, float, bool)) else ''}"
            scalars = []

            def _collect(x):
                if torch.is_tensor(x) and x.numel() == 1:
                    scalars.append(float(x))
                elif isinstance(x, tuple):
                    for y in x:
                        _collect(y)
            _collect(got)
            if scalars:
                vals = ", ".join(f"{v:.6e}" for v in scalars)
                best = min(scalars, key=lambda v: abs(v - ours_I))
                rel = abs(best - ours_I) / max(abs(ours_I), 1e-12)
                log("B compute_importance", "MATCH" if rel < 1e-5 else "MISMATCH",
                    f"type={ok_cfg} structure={_flat(got)} scalars=[{vals}] "
                    f"ours avg|W*G|={ours_I:.6e} closest rel={rel:.3e}")
            else:
                log("B compute_importance", "CHECK", f"type={ok_cfg} structure={_flat(got)} (no scalars)")

        # ---------------------------------------------------------- E. dynamic_init (init B)
        try:
            with torch.no_grad():
                wp.copy_(W)
            wp.grad = G.clone()
            wp.grad_stored = G.clone()
            before = set(dir(layer))
            layer.dynamic_init(avg_rank=R_REF, rank=R_REF, scaling_by_lr=False)
            new_attrs = [a for a in dir(layer) if a not in before and not a.startswith("_")]
            wa = getattr(layer, "weight_a", None)
            wb = getattr(layer, "weight_b", None)
            shp = lambda t: tuple(t.shape) if torch.is_tensor(t) else type(t).__name__
            log("E dynamic_init", "OK",
                f"new attrs={new_attrs} weight_a={shp(wa)} weight_b={shp(wb)} "
                f"avg_lora_rank={getattr(layer, 'avg_lora_rank', None)} scale_rank={getattr(layer, 'scale_rank', None)}")
            # author dW vs our gora_init dW on the SAME A and G
            if torch.is_tensor(wa) and torch.is_tensor(wb):
                from lorastack.adapters.gora_init import _b0_solve, _xi
                import math as _math
                # orientation: try (out,r)@(r,in) vs (r,out)@(in,r) — pick the (M,N)-shaped one
                cand = []
                for fa, fb in ((wa, wb), (wb, wa)):
                    d = fb @ fa if fa.dim() == 2 and fb.dim() == 2 else None
                    if d is not None and tuple(d.shape) == (M, N):
                        cand.append(d)
                scaler = float(getattr(layer, "lora_scaler", 1.0))
                dW_auth = cand[0] * scaler if cand else None
                if dW_auth is not None:
                    # our formula with the SAME A (author's weight_a transposed as needed) and s=1
                    A_ours = wa if tuple(wa.shape) == (R_REF, N) else wa.t()
                    B0 = _b0_solve(G, A_ours.detach().float())
                    s_ = 1.0
                    B0 *= _xi(0.05, s_, R_REF, N)  # gamma=0.05 (our paper table r=8), s=1
                    dW_ours = B0 @ A_ours.detach().float()
                    num = float((dW_auth - dW_ours).pow(2).sum())
                    cos = float(torch.nn.functional.cosine_similarity(
                        dW_auth.flatten(), dW_ours.flatten(), dim=0))
                    c = float((dW_auth * dW_ours).sum() / max(float((dW_ours * dW_ours).sum()), 1e-30))
                    log("E dW author vs ours", "MATCH" if cos > 0.999 else "CHECK",
                        f"cosine={cos:.6f} scale_fit c={c:.4f} rel_l2={(num / max(float(dW_ours.pow(2).sum()), 1e-30)) ** 0.5:.4f} "
                        f"(author lora_scaler={scaler})")
        except Exception as e:
            log("E dynamic_init", "DEFERRED", f"{type(e).__name__}: {str(e)[:200]}")

        # ---------------------------------------------------------- C. get_allocated_rank
        author_ranks, author_imps = None, None
        try:
            # tiny model with TWO author layers so allocation spreads a budget
            torch.manual_seed(SEED)
            model = torch.nn.Sequential(
                gora.LinearWithGoRA(_construct_probe(gora.LoRAConfig, dict(
                    lora_rank=R_REF, lora_scaler=1.0, in_features=N, out_features=M))),
                torch.nn.ReLU(),
                gora.LinearWithGoRA(_construct_probe(gora.LoRAConfig, dict(
                    lora_rank=R_REF, lora_scaler=1.0, in_features=M, out_features=16))),
            )
            torch.manual_seed(SEED + 1)
            x = torch.randn(4, N)
            for p in model.parameters():
                p.requires_grad_(True)
            model(x).sum().backward()
            # author convention: per-weight stored gradient (hook product == dL/dW here)
            for mod in model:
                w = mod.weight if hasattr(mod, "weight") else None
                if w is not None and w.grad is not None:
                    w.grad_stored = w.grad.detach().clone()
            args = SimpleNamespace(
                lora_rank=R_REF, gora_max_rank=R_MAX, gora_min_rank=R_MIN,
                gora_importance_type="union_mean", gora_scale_importance=False,
            )
            res = None
            for _ in range(15):
                try:
                    res = gora.get_allocated_rank(model, args)
                    break
                except AttributeError as e:
                    m = re.search(r"attribute '(\w+)'", str(e))
                    if not m:
                        raise
                    key = m.group(1)
                    args.__setattr__(key, dict(gora_allocate_stretagy="moderate",
                                               gora_features_func=None).get(key, 1.0))
            if res is None:
                log("C get_allocated_rank", "DEFERRED", "args filling exhausted")
            else:
                log("C get_allocated_rank", "OK", f"return signature len={len(res)}; args used={sorted(vars(args))}")
                if len(res) == 4:
                    total_budget, actual_trainable, named_ranks, named_importances = res
                    author_ranks = dict(named_ranks)
                    author_imps = dict(named_importances)
                    log("C author ranks", "OK", f"total_budget={total_budget} trainable={actual_trainable} ranks={author_ranks}")
        except Exception as e:
            log("C get_allocated_rank", "DEFERRED", f"{type(e).__name__}: {str(e)[:200]}")

        # ---------------------------------------------------------- D. our phase (b), same inputs
        try:
            from lorastack.adapters.gora_alloc import _allocate_ranks_from_importance
            if author_imps is not None:
                shapes = {}
                for name, mod in model.named_modules():
                    if isinstance(mod, gora.LinearWithGoRA):
                        w = mod.weight if hasattr(mod, "weight") else mod.base_layer.weight
                        shapes[name] = (int(w.shape[0]), int(w.shape[1]))
                imps = {k: float(v) if torch.is_tensor(v) and v.numel() == 1 else float(torch.tensor(v).float().mean())
                        for k, v in author_imps.items()}
                cfg = SimpleNamespace(rank=R_REF, gora_r_min=None, gora_r_max=None)
                our_ranks = _allocate_ranks_from_importance(imps, shapes, cfg)
                log("D our phase(b) on author importances", "OK",
                    f"imps={ {k: round(v, 6) for k, v in imps.items()} } shapes={shapes} -> ranks={our_ranks}")
                keys = set(author_ranks) & set(our_ranks)
                if keys:
                    match = all(author_ranks[k] == our_ranks[k] for k in keys)
                    log("C-vs-D ranks", "MATCH" if match else "MISMATCH",
                        f"author={ {k: author_ranks[k] for k in sorted(keys)} } "
                        f"ours={ {k: our_ranks[k] for k in sorted(keys)} }")
                else:
                    log("C-vs-D ranks", "MISMATCH", f"no common keys author={author_ranks} ours={our_ranks}")
            else:
                log("D our phase(b)", "DEFERRED", "no author importances (C failed)")
        except Exception as e:
            log("D our phase(b)", "DEFERRED", f"{type(e).__name__}: {str(e)[:200]}")

        # ---------------------------------------------------------- P. rounding discriminators
        # author gave 12 where our round-half-up gives 13 (raw=12.74). Probe the author's
        # rounding rule by feeding scaled G (importance scales linearly) so the raw value
        # of layer '2' lands exactly on .5 boundaries; compare with our rules.
        try:
            from lorastack.adapters.gora_alloc import _allocate_ranks_from_importance
            import math

            def author_ranks_scaled(scale2):
                torch.manual_seed(SEED)
                mdl = torch.nn.Sequential(
                    gora.LinearWithGoRA(gora.LoRAConfig(lora_rank=R_REF, lora_scaler=1.0, in_features=N, out_features=M)),
                    torch.nn.ReLU(),
                    gora.LinearWithGoRA(gora.LoRAConfig(lora_rank=R_REF, lora_scaler=1.0, in_features=M, out_features=16)),
                )
                torch.manual_seed(SEED + 1)
                xx = torch.randn(4, N)
                for p in mdl.parameters():
                    p.requires_grad_(True)
                mdl(xx).sum().backward()
                mods = [m for m in mdl if isinstance(m, gora.LinearWithGoRA)]
                for i, m_ in enumerate(mods):
                    w_ = m_.weight if hasattr(m_, "weight") else m_.base_layer.weight
                    gs = w_.grad.detach().clone()
                    if i == 1:
                        gs = gs * scale2
                    w_.grad_stored = gs
                args2 = SimpleNamespace(lora_rank=R_REF, gora_max_rank=R_MAX, gora_min_rank=R_MIN,
                                        gora_importance_type="union_mean", gora_scale_importance=False,
                                        gora_allocate_stretagy="moderate", gora_features_func=None)
                res2 = None
                for _ in range(15):
                    try:
                        res2 = gora.get_allocated_rank(mdl, args2)
                        break
                    except AttributeError as e2:
                        m2 = re.search(r"attribute '(\w+)'", str(e2))
                        if not m2:
                            raise
                        args2.__setattr__(m2.group(1), False)
                if res2 is None:
                    raise RuntimeError("args filling exhausted in P")
                return dict(res2[2]), mods

            def predict(imps0, imps1, shapes_, rule):
                cfgp = SimpleNamespace(rank=R_REF, gora_r_min=None, gora_r_max=None)
                if rule == "floor":
                    # emulate floor by monkey-rounded copy of the allocator math
                    total = imps0 + imps1
                    (m0, n0), (m1, n1) = shapes_["0"], shapes_["2"]
                    B_tot = ((m0 + n0) + (m1 + n1)) * R_REF
                    r0 = int(min(32, max(4, math.floor(B_tot * imps0 / total / (m0 + n0)))))
                    r1 = int(min(32, max(4, math.floor(B_tot * imps1 / total / (m1 + n1)))))
                    return {"0": r0, "2": r1}
                ranks = _allocate_ranks_from_importance({"0": imps0, "2": imps1}, shapes_, cfgp)
                return ranks

            shapes_ = {"0": (M, N), "2": (16, M)}
            for target in (12.5, 14.5):
                # closed-form scale so OUR raw value for layer '2' hits `target`
                r0, r1 = author_ranks_scaled(1.0)  # warm run for baseline importances
                # recompute imps from the baseline stored grads
                w0 = r1[0].weight if hasattr(r1[0], "weight") else r1[0].base_layer.weight
                w1 = r1[1].weight if hasattr(r1[1], "weight") else r1[1].base_layer.weight
                i0_ = float((w0.detach() * w0.grad_stored).abs().mean())
                i1_ = float((w1.detach() * w1.grad_stored).abs().mean())
                B_tot = ((M + N) + (16 + M)) * R_REF
                A_t = target * (16 + M) / B_tot
                x = A_t * i0_ / (1 - A_t)
                s = x / i1_
                ranks_a, _ = author_ranks_scaled(s)
                pred_half = predict(i0_, s * i1_, shapes_, "half")
                pred_floor = predict(i0_, s * i1_, shapes_, "floor")
                log(f"P raw={target}", "OK",
                    f"scale={s:.4f} author={ranks_a} our_half_up={pred_half} our_floor={pred_floor}")
        except Exception as e:
            log("P rounding discriminators", "DEFERRED", f"{type(e).__name__}: {str(e)[:200]}")

print("\n===== V2 SUMMARY =====")
for tag, status, detail in report:
    print(f"[{status:9s}] {tag:52s} {str(detail)[:150]}")


