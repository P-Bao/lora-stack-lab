"""V3 black-box: AltLoRA optimizer_new.py + BA-LoRA train.py (author code) vs lorastack.

Isolation: this script itself runs as its own process (nothing installed, nothing written
outside tests/blackbox); author modules are imported via sys.path only. Their peft fork is
NEVER allowed to shadow the local peft — import probes with the fork prepended run in a
separate `python -c` subprocess and only report OK/exit-reason.

What was READ from author code (disclosed): grep of def/class lines in optimizer_new.py and
train.py (signatures only). No algorithm bodies studied.

Run:  python tests/blackbox/run_altlora_balora_blackbox.py
"""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

ALTLORA_DIR = Path(r"D:\Documents\NCKH\NLP\ReaserchTeamNLP\LoRA\AltLoRA\AltLoRA")
ALTLORA_PEFT = ALTLORA_DIR / "peft"
BALORA_DIR = Path(r"D:\Documents\NCKH\NLP\ReaserchTeamNLP\LoRA\BA-LoRA\BA-LoRA")

SEED = 0
R, DIN, DOUT = 8, 32, 64

report = []


def log(tag, status, detail=""):
    report.append((tag, status, detail))
    print(f"[{status:9s}] {tag:52s} {str(detail)[:170]}")


def probe_import(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    ok = r.returncode == 0 and "IMPORT_OK" in r.stdout
    tail = (r.stderr or "").strip().splitlines()
    return ok, (r.stdout.strip().splitlines() or [""])[-1] if ok else (tail[-1] if tail else "exit " + str(r.returncode))


# ---------------------------------------------------------------- A. import probes (subprocess)
ok_plain, msg_plain = probe_import([
    sys.executable, "-c",
    f"import sys; sys.path.insert(0, r'{ALTLORA_DIR}'); import optimizer_new; print('IMPORT_OK')"])
log("A altlora import (subprocess)", "OK" if ok_plain else "DEFERRED", msg_plain)

ok_fork, msg_fork = probe_import([
    sys.executable, "-c",
    f"import sys; sys.path.insert(0, r'{ALTLORA_PEFT}'); sys.path.insert(0, r'{ALTLORA_DIR}'); "
    "import optimizer_new; print('IMPORT_OK')"])
log("A' altlora import (+their peft fork)", "OK" if ok_fork else "DEFERRED", msg_fork)

ok_bal, msg_bal = probe_import([
    sys.executable, "-c",
    f"import sys; sys.path.insert(0, r'{BALORA_DIR}'); import train; print('IMPORT_OK')"])
log("A'' balora train import (subprocess)", "OK" if ok_bal else "DEFERRED", msg_bal)

# ---------------------------------------------------------------- B. AltLoRA 1-step compare
if ok_plain:
    sys.path.insert(0, str(ALTLORA_DIR))
    try:
        import optimizer_new  # noqa: E402
        from lorastack.optim.altlora import AltLoRAOptimizer, _collect_pairs

        torch.manual_seed(SEED)
        gg = torch.Generator().manual_seed(SEED)

        import torch.nn as nn

        def make_pair_model():
            """PEFT-like nesting: params named lora_A.default.weight / lora_B.default.weight."""

            class P(nn.Module):
                def __init__(self):
                    super().__init__()
                    a = nn.Linear(DIN, R, bias=False)
                    b = nn.Linear(R, DOUT, bias=False)
                    self.lora_A = nn.ModuleDict({"default": a})
                    self.lora_B = nn.ModuleDict({"default": b})

            m = P()
            return m

        def set_params(m, A0, B0):
            with torch.no_grad():
                m.lora_A["default"].weight.copy_(A0)
                m.lora_B["default"].weight.copy_(B0)

        def set_grads(m, gA, gB):
            m.lora_A["default"].weight.grad = gA.clone()
            m.lora_B["default"].weight.grad = gB.clone()

        A0 = torch.randn(R, DIN, generator=gg) * 0.1
        B0 = torch.randn(DOUT, R, generator=gg) * 0.1
        gA = torch.randn(R, DIN, generator=gg) * 0.1
        gB = torch.randn(DOUT, R, generator=gg) * 0.1
        LR, WD, REG = 1e-3, 1e-5, 1e-3

        # author optimizer
        am = make_pair_model()
        set_params(am, A0, B0)
        set_grads(am, gA, gB)
        try:
            aopt = optimizer_new.altlora(am, lr=LR, betas=(0.999, 0.98), eps=1e-6,
                                         weight_decay=WD, correct_bias=True, rank=2, reg=REG)
            n_auth_params = sum(len(gp["params"]) for gp in aopt.param_groups)
            aopt.step()
            wA = am.lora_A["default"].weight
            wB = am.lora_B["default"].weight
            dA_auth = (wA.data - A0).clone()
            dB_auth = (wB.data - B0).clone()
            log("B author altlora step1", "OK",
                f"collected_params={n_auth_params} |dB|={float(dB_auth.norm()):.6e} "
                f"|dA|={float(dA_auth.norm()):.6e}")
        except Exception as e:
            log("B author altlora step1", "DEFERRED", f"{type(e).__name__}: {str(e)[:220]}")
            aopt = None
            dA_auth = dB_auth = None

        # our optimizer, same inputs
        om = make_pair_model()
        set_params(om, A0, B0)
        set_grads(om, gA, gB)
        cfg = SimpleNamespace(alpha=1.0, rank=1)  # module lacks .scaling -> s = 1.0
        pairs = _collect_pairs(cfg, om)
        oopt = AltLoRAOptimizer(pairs, [om.lora_A["default"].weight, om.lora_B["default"].weight],
                                lr=LR, beta1=0.999, weight_decay=WD, ridge=REG, order="B_first")
        oopt.step()
        wB_ours = om.lora_B["default"].weight
        dB_ours = (wB_ours.data - B0).clone()

        if dA_auth is not None:
            if float(dA_auth.norm()) == 0.0 and float(dB_auth.norm()) > 0.0:
                # author also alternates: step1 moves B only -> direct compare
                denom = max(float(dB_ours.norm()), 1e-30)
                rel = float((dB_auth - dB_ours).norm()) / denom
                cos = float(torch.nn.functional.cosine_similarity(dB_auth.flatten(), dB_ours.flatten(), dim=0))
                log("B step1 dB compare", "MATCH" if rel < 1e-5 else "CHECK",
                    f"rel_l2={rel:.4e} cosine={cos:.6f} |dB|_auth={float(dB_auth.norm()):.6e} "
                    f"|dB|_ours={float(dB_ours.norm()):.6e}")
                # step 2: A phase
                set_grads(am, gA, gB)
                set_grads(om, gA, gB)
                aopt.step()
                oopt.step()
                dA_ours = om.lora_A["default"].weight.data - A0
                dA_auth_tot = am.lora_A["default"].weight.data - A0
                denom = max(float(dA_ours.norm()), 1e-30)
                rel = float((dA_auth_tot - dA_ours).norm()) / denom
                cos = float(torch.nn.functional.cosine_similarity(dA_auth_tot.flatten(), dA_ours.flatten(), dim=0))
                log("B step2 dA compare", "MATCH" if rel < 1e-5 else "CHECK",
                    f"rel_l2={rel:.4e} cosine={cos:.6f} |dA|_auth={float(dA_auth_tot.norm()):.6e} "
                    f"|dA|_ours={float(dA_ours.norm()):.6e}")
            elif float(dB_auth.norm()) == 0.0 and float(dA_auth.norm()) == 0.0:
                log("B step1 dB compare", "DEFERRED",
                    "author optimizer collected the model but updated nothing in step1 "
                    "(param collection/structure mismatch — needs author-side integration, Kaggle)")
            else:
                log("B step1 dB compare", "CHECK",
                    f"author updated BOTH factors in one step (|dA|={float(dA_auth.norm()):.3e}) "
                    f"-> structural difference vs our alternating phases; |dB|_auth={float(dB_auth.norm()):.6e} "
                    f"|dB|_ours={float(dB_ours.norm()):.6e}")

            # probe: author with rank=R (matching factor rank) + per-step trace
            try:
                am2 = make_pair_model()
                set_params(am2, A0, B0)
                set_grads(am2, gA, gB)
                aopt2 = optimizer_new.altlora(am2, lr=LR, betas=(0.999, 0.98), eps=1e-6,
                                              weight_decay=WD, correct_bias=True, rank=R, reg=REG)
                trace = []
                wA2, wB2 = am2.lora_A["default"].weight, am2.lora_B["default"].weight
                for step in range(3):
                    set_grads(am2, gA, gB)  # constant gradient (linear loss probe)
                    aopt2.step()
                    trace.append((float((wA2.data - A0).norm()), float((wB2.data - B0).norm())))
                log("B' author trace(rank=R)", "OK",
                    "step(|dA|,|dB|): " + " -> ".join(f"({a:.3e},{b:.3e})" for a, b in trace))
            except Exception as e:
                log("B' author trace", "DEFERRED", f"{type(e).__name__}: {str(e)[:200]}")
    except Exception as e:
        log("B AltLoRA compare", "DEFERRED", f"{type(e).__name__}: {str(e)[:220]}")

# ---------------------------------------------------------------- C. BA-LoRA regularizer probe
if ok_bal:
    try:
        sys.path.insert(0, str(BALORA_DIR))
        import train as balora_train  # noqa: E402
        fn = getattr(balora_train.BALoRATrainer, "_randomized_topk_singular_values", None)
        if fn is None:
            log("C balora randomized topk", "DEFERRED", "attribute not found on BALoRATrainer")
        else:
            torch.manual_seed(SEED)
            g2 = torch.Generator().manual_seed(SEED)
            mat = torch.randn(40, 5, generator=g2) @ torch.randn(5, 16, generator=g2)  # rank-5
            try:
                out = fn(mat, 5)  # unbound call — works for @staticmethod
                out_t = out if torch.is_tensor(out) else torch.tensor(out)
                s_exact = torch.linalg.svdvals(mat.float())[: out_t.numel()]
                vals_auth = torch.sort(out_t.flatten(), descending=True).values
                rel = float((vals_auth - s_exact).norm() / s_exact.norm())
                from lorastack.losses.balora_reg import _svdr_exact
                ours_ratio = -float(_svdr_exact(mat, 5))
                log("C balora _randomized_topk_singular_values", "MATCH" if rel < 0.05 else "CHECK",
                    f"rel_vs_exact_svals={rel:.4e} author={ [round(float(v), 3) for v in vals_auth] } "
                    f"exact={ [round(float(v), 3) for v in s_exact] } our _svdr_exact(top5 mass)={ours_ratio:.6f}")
            except TypeError as e:
                log("C balora randomized topk", "DEFERRED",
                    f"not callable unbound (needs instance/other signature): {str(e)[:160]}")
    except Exception as e:
        log("C balora probe", "DEFERRED", f"{type(e).__name__}: {str(e)[:220]}")

print("\n===== V3 SUMMARY =====")
for tag, status, detail in report:
    print(f"[{status:9s}] {tag:52s} {str(detail)[:150]}")
