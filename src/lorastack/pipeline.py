"""Build model + components in the FIXED order of plan.md section 4:

    rank allocation (O) -> adapter build (L|G) -> init (Oi|PiSSA) -> optimizer (A) -> loss (B/Bp/Br)

fp16/T4: base model fp16, autocast(fp16) + GradScaler; adapter params fp32 (PEFT default keeps
new params in fp32 when base is fp16). Inverse/projection math inside reimpl modules must use fp32.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import torch
from peft import GraloraConfig, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapters.base import AdapterContext
from .config import ExpConfig
from .data.loaders import make_gora_probe_loader, make_train_loader
from .eval import evaluate
from .registry import get_hooks
from .utils.ckpt import load_ckpt, save_ckpt
from .utils.env_report import env_report
from .utils.logging import JsonlLogger
from .utils.seed import set_seed

B_CODES = {"B", "Bp", "Br"}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=10).strip()
    except Exception:
        return "unknown"


def build(cfg: ExpConfig):
    """Returns (model, tokenizer, ctx, optimizer). Enforces X1 (Oi vs PiSSA mutual exclusion)."""
    set_seed(cfg.seed)
    components = list(cfg.components)

    if "Oi" in components and "Bp" in components:  # X1
        raise ValueError("X1: Oi (GoRA init) and Bp (PiSSA init) are mutually exclusive; use TETRAD-P or TETRAD-R")
    if "O" in components and "Oi" not in components and not ({*components} & (B_CODES | {"A", "L", "G"})):
        raise ValueError("O alone has no effect; pair it with L/G")

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if cfg.fp16 else torch.float32,
        attn_implementation="sdpa",  # T4: no flash-attn
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ctx = AdapterContext(exp_id=cfg.exp_id, components=components, rank=cfg.rank)

    # 1. rank allocation (O): streaming N-batch gradient pre-pass on the BASE model
    if "O" in components or "Oi" in components:
        ctx.probe_loader = make_gora_probe_loader(cfg, tok, cfg.gora_n_batches)
    if "O" in components:
        ctx.layer_ranks = get_hooks("O").allocate_ranks(cfg, model, tok, ctx.probe_loader)

    # 2. adapter build (L | G)
    if "G" in components:
        if cfg.rank % cfg.gralora_k != 0:
            raise ValueError(f"GraloraConfig requires r % gralora_k == 0, got r={cfg.rank}, k={cfg.gralora_k}")
        peft_cfg = GraloraConfig(r=cfg.rank, gralora_k=cfg.gralora_k, alpha=cfg.alpha,
                                 target_modules=cfg.target_modules, task_type="CAUSAL_LM")
    else:
        peft_cfg = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, target_modules=cfg.target_modules,
                              task_type="CAUSAL_LM")
    if "O" in components:
        get_hooks("O").patch_config(cfg, peft_cfg, ctx)  # L: rank_pattern; G: group-budget + deviation
    if "Bp" in components:
        if isinstance(peft_cfg, GraloraConfig):
            # GraloraConfig has no init_lora_weights (peft_inventory) — PiSSA init for G layers
            # is NOT available via PEFT; stage 5 must decide: custom pissa-for-gralora module or
            # documented deviation. Fail loud instead of silently losing the init (X1).
            raise NotImplementedError(
                "Bp (PiSSA init) + G: GraloraConfig has no init_lora_weights in peft "
                f"{__import__('peft').__version__}; needs custom pissa_init for GraLoRA (see decisions.md D-X1b)"
            )
        peft_cfg.init_lora_weights = "pissa"  # PEFT public API; modifies W_res (X1)
        ctx.init_mode = "pissa"
    model = get_peft_model(model, peft_cfg)
    if "G" in components:
        ctx.gralora_k = cfg.gralora_k

    # 3. init (Oi) — after adapter exists; never together with pissa (X1)
    if "Oi" in components:
        get_hooks("Oi").init_adapter(cfg, model, ctx)
        ctx.init_mode = "gora"

    # 4. optimizer (A | AdamW)
    if "A" in components:
        optimizer = get_hooks("A").make_optimizer(cfg, model, ctx)
    else:
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    # collect (A,B) pairs for AltLoRA block-wise projection if it was not done inside make_optimizer
    if "A" in components and not ctx.ab_pairs:
        _collect_ab_pairs(model, ctx)
    return model, tok, ctx, optimizer


def _collect_ab_pairs(model, ctx) -> None:
    """Pairs (lora_A.weight, lora_B.weight) per adapted module; for GraLoRA one entry per sub-block."""
    pairs = []
    for name, mod in model.named_modules():
        params = dict(mod.named_parameters(recurse=False))
        a_keys = sorted(k for k in params if ".lora_A." in k or k.startswith("gralora_A"))
        b_keys = sorted(k for k in params if ".lora_B." in k or k.startswith("gralora_B"))
        if a_keys and b_keys:
            pairs.extend((params[a], params[b], name) for a, b in zip(a_keys, b_keys))
    ctx.ab_pairs = pairs


def base_forward(model, batch):
    """Forward with adapters disabled (= base model). Note: for PiSSA the stored base weight is
    W_res, not W0 — loss hooks must consult spec/balora.md for the correct reference."""
    with model.disable_adapter():
        return model(**batch)


def train(cfg: ExpConfig, model, tok, ctx, optimizer) -> dict:
    guard = time.time() + cfg.timeguard_minutes * 60
    logger = JsonlLogger(str(Path(cfg.out_dir) / f"{cfg.exp_id}.log.jsonl"))
    loader = make_train_loader(cfg, tok)
    scaler = torch.amp.GradScaler(enabled=cfg.fp16)
    loss_codes = [c for c in cfg.components if c in B_CODES]
    loss_hooks = [get_hooks(c) for c in loss_codes] if loss_codes else []

    start_step = load_ckpt(cfg.ckpt_dir, cfg.exp_id, model, optimizer) or 0
    steps_per_epoch = len(loader) // cfg.grad_accum
    total_steps = cfg.max_steps if cfg.max_steps > 0 else int(steps_per_epoch * cfg.epochs)
    ctx.total_steps, ctx.step = total_steps, start_step
    global_step, curve, t0 = start_step, [], time.time()
    model.train()

    data_iter = iter(loader)
    running = 0.0
    while global_step < total_steps:
        if time.time() > guard:  # TimeGuard: checkpoint + clean stop, no eval
            save_ckpt(cfg.ckpt_dir, cfg.exp_id, model, optimizer, global_step)
            return {"status": "timeguard", "step": global_step, "train_loss_curve": curve,
                    "resumed_from": start_step or None}
        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.fp16):
                out = model(**batch)
                loss = out.loss / cfg.grad_accum
                for h in loss_hooks:  # 5. loss (B/Bp/Br): weighted regularizers
                    terms = h.loss_terms(cfg, model, ctx, batch, lambda b: base_forward(model, b))
                    for v in terms.values():
                        loss = loss + v / cfg.grad_accum
            scaler.scale(loss).backward()
            running += loss.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        global_step += 1
        ctx.step = global_step

        if global_step % 10 == 0:
            curve.append({"step": global_step, "loss": round(running / cfg.grad_accum, 4)})
            logger.log({"step": global_step, "loss": running / cfg.grad_accum})
            running = 0.0
        if global_step % cfg.ckpt_every_steps == 0:
            save_ckpt(cfg.ckpt_dir, cfg.exp_id, model, optimizer, global_step)

    metrics = evaluate(model, tok, cfg, model.device) if time.time() < guard else {}
    return {"status": "done", "step": global_step, "train_loss_curve": curve,
            "metrics": metrics, "wall_time_s": round(time.time() - t0, 1),
            "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0,
            "resumed_from": start_step or None}


def run_experiment(cfg: ExpConfig) -> dict:
    t0 = time.time()
    model, tok, ctx, optimizer = build(cfg)
    result = train(cfg, model, tok, ctx, optimizer)
    result.update({
        "id": cfg.exp_id,
        "git_commit": _git_commit(),
        "seed": cfg.seed,
        "impl": {c: ("peft" if c in ("L", "G") else "reimpl+peft_init" if c == "Bp" else "reimpl")
                 for c in cfg.components},
        "init_mode": ctx.init_mode,
        "lr": cfg.lr,
        "rank": cfg.rank,
        "k": cfg.gralora_k if "G" in cfg.components else None,
        "rank_per_layer": ctx.layer_ranks,
        "rank_budget_deviation": ctx.rank_budget_deviation,
        "config": cfg.dump(str(Path(cfg.out_dir) / f"{cfg.exp_id}.config.yaml")),
        "env": env_report(),
        "total_wall_time_s": round(time.time() - t0, 1),
    })
    return result
