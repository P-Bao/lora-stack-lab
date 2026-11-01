"""Evaluators (lean): GSM8K exact-match via last-number extraction; GLUE verbalized accuracy.
Absolute scores on 0.5B are low by design — the study compares configs relative to L* (plan.md Gate 1)."""
from __future__ import annotations

import re

import torch

from ..data.loaders import load_eval_prompts


def _last_number(text: str) -> str | None:
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return matches[-1] if matches else None


@torch.no_grad()
def eval_gsm8k(model, tokenizer, cfg, device) -> dict:
    rows = load_eval_prompts("gsm8k", cfg.data.eval_size)
    correct = 0
    for r in rows:
        prompt = f"Question: {r['q']}\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        out = model.generate(**ids, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = _last_number(text)
        gold = _last_number(r["gold"])
        correct += int(pred is not None and gold is not None and abs(float(pred) - float(gold)) < 1e-4)
    return {"gsm8k_acc": correct / len(rows), "n_eval": len(rows)}


@torch.no_grad()
def eval_glue(model, tokenizer, cfg, device) -> dict:
    rows = load_eval_prompts(cfg.data.eval_name, cfg.data.eval_size)
    names = ["negative", "positive"]
    correct = 0
    for r in rows:
        prompt = f"Sentence: {r['q']}\nSentiment:"
        best, best_logp = None, -1e9
        for cand in names:
            full = tokenizer(prompt + f" {cand}", return_tensors="pt").to(device)
            out = model(**full)
            # logprob of the candidate token(s)
            lp = 0.0
            ids = full["input_ids"][0]
            logits = out.logits[0]
            for pos in range(len(ids) - len(tokenizer(f" {cand}", add_special_tokens=False)["input_ids"]), len(ids) - 1):
                lp += torch.log_softmax(logits[pos], dim=-1)[ids[pos + 1]].item()
            if lp > best_logp:
                best_logp, best = lp, names.index(cand)
        correct += int(best == r["gold"])
    return {f"{cfg.data.eval_name}_acc": correct / len(rows), "n_eval": len(rows)}


def evaluate(model, tokenizer, cfg, device) -> dict:
    if cfg.data.eval_name == "gsm8k":
        return eval_gsm8k(model, tokenizer, cfg, device)
    if cfg.data.eval_name.startswith("glue:"):
        return eval_glue(model, tokenizer, cfg, device)
    raise ValueError(f"no evaluator for {cfg.data.eval_name}")
