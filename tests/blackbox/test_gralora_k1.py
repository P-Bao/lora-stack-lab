"""V1 black-box: GraLoRA with gralora_k=1 must degenerate to plain LoRA (upstream peft 0.20.0).

Checks on a fixed-seed tiny nn.Linear stack:
  1. parameter shapes: gralora_A/B[0] == lora_A/lora_B (k=1 -> one full block)
  2. init Delta-W: both exactly zero (PEFT B=0 default) and same A init scale (kaiming)
  3. identical hand-assigned A/B -> identical Delta-W and identical forward output
Author code is NOT used here: GraloraConfig ships in upstream peft (HF team), verified via
pip metadata (Home-page github.com/huggingface/peft).

Run:  python tests/blackbox/test_gralora_k1.py   (or pytest tests/blackbox/test_gralora_k1.py)
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from peft import GraloraConfig, LoraConfig, get_peft_model  # noqa: E402

SEED = 0
R = 8
K = 1
TOL_FORWARD = 1e-5   # fp32 CPU, report actual below
TOL_DELTAW = 1e-5
TOL_SCALE = 1e-12


def _base_model():
    torch.manual_seed(SEED)
    return torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 16))


def _adapt(base, cfg):
    torch.manual_seed(SEED)  # same init seed for both variants
    return get_peft_model(base, cfg)


def _tuner_layer(model, idx):
    return model.base_model.model[idx]


def main():
    results = []

    base_a = _base_model()
    base_b = _base_model()
    assert torch.equal(base_a[0].weight, base_b[0].weight), "base models differ"

    lora_model = _adapt(base_a, LoraConfig(r=R, lora_alpha=8, target_modules=["0", "2"]))
    gra_model = _adapt(base_b, GraloraConfig(r=R, gralora_k=K, alpha=8, target_modules=["0", "2"]))

    # --- 1. shapes ---------------------------------------------------------
    for idx in (0, 2):
        lora_layer = _tuner_layer(lora_model, idx)
        gra_layer = _tuner_layer(gra_model, idx)
        a_lora = lora_layer.lora_A["default"].weight  # Parameter (r, in)
        b_lora = lora_layer.lora_B["default"].weight  # (out, r)
        a_gra = gra_layer.gralora_A["default"][0]     # (k, in_blk, r) -> [0] = (in_blk, r)
        b_gra = gra_layer.gralora_B["default"][0]     # (k, r, out_blk) -> [0] = (r, out_blk)
        in_f, out_f = int(lora_layer.in_features), int(lora_layer.out_features)
        # gralora stores (in_blk, r) / (r, out_blk) — TRANSPOSED vs lora (r, in) / (out, r)
        shape_ok = (a_gra.shape == a_lora.shape[::-1]) and (b_gra.shape == b_lora.shape[::-1])
        results.append((f"L{idx} shapes k=1 == lora (transposed)", shape_ok,
                        f"gralora_A[0]{tuple(a_gra.shape)} vs lora_A{tuple(a_lora.shape)}, "
                        f"gralora_B[0]{tuple(b_gra.shape)} vs lora_B{tuple(b_lora.shape)}"))

        # --- 2. init Delta-W and A scale -----------------------------------
        # dW = B @ A.T for lora (b: out×r, a: r×in); gralora stores transposed:
        # b_gra (r, out), a_gra (in, r) -> dW = b_gra.T @ a_gra.T
        dw_lora = b_lora.data @ a_lora.data
        dw_gra = b_gra.data.t() @ a_gra.data.t()
        zero_ok = float(dw_lora.abs().max()) == 0.0 and float(dw_gra.abs().max()) == 0.0
        std_ratio = float(a_gra.data.std() / a_lora.data.std())
        results.append((f"L{idx} init dW == 0 (both)", zero_ok,
                        f"max|dW| lora={float(dw_lora.abs().max()):.3e} gra={float(dw_gra.abs().max()):.3e}"))
        results.append((f"L{idx} A-init std ratio gra/lora", 0.8 <= std_ratio <= 1.25,
                        f"std_gra={float(a_gra.data.std()):.4f} std_lora={float(a_lora.data.std()):.4f} ratio={std_ratio:.3f}"))

        # --- 3. identical hand-assigned A/B -> identical dW and forward ------
        g = torch.Generator().manual_seed(123)
        A = torch.randn(R, in_f, generator=g) * 0.1
        B = torch.randn(out_f, R, generator=g) * 0.1
        with torch.no_grad():
            a_lora.copy_(A)
            b_lora.copy_(B)
            a_gra.copy_(A.t())   # gralora_A stores (in_blk, r) — transposed vs lora_A
            b_gra.copy_(B.t())   # gralora_B stores (r, out_blk)
        dw_lora = b_lora.data @ a_lora.data
        dw_gra = b_gra.data.t() @ a_gra.data.t()
        d_max = float((dw_lora - dw_gra).abs().max())
        results.append((f"L{idx} hand dW identical", d_max <= TOL_DELTAW, f"max|dW_l - dW_g|={d_max:.3e}"))

        s_lora = float(lora_layer.scaling["default"])
        s_gra = float(gra_layer.scaling["default"])
        results.append((f"L{idx} scaling equal", abs(s_lora - s_gra) <= TOL_SCALE,
                        f"s_lora={s_lora} s_gra={s_gra}"))

        # forward equality (dropout 0, eval mode)
        lora_model.eval(); gra_model.eval()
        # stack input = layer-0 in_features (32); forward runs the whole stack
        x = torch.randn(4, 32, generator=g)
        with torch.no_grad():
            y_lora = lora_model(x)
            y_gra = gra_model(x)
        f_max = float((y_lora - y_gra).abs().max())
        rel = f_max / max(float(y_lora.abs().max()), 1e-12)
        results.append((f"L-stack forward identical", f_max <= TOL_FORWARD,
                        f"max|y_l - y_g|={f_max:.3e} rel={rel:.3e}"))

    print(f"{'CHECK':44s}  RESULT  DETAIL")
    all_ok = True
    for name, ok, detail in results:
        all_ok &= bool(ok)
        print(f"{name:44s}  {'PASS' if ok else 'FAIL'}    {detail}")
    print("V1 OVERALL:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())




