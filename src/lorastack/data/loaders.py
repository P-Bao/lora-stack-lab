"""Dataset loaders: fixed subsets (seed 42) shared by every stage (plan.md section 7, stage 1)."""
from __future__ import annotations

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

SUBSET_SEED = 42


def _load_raw(name: str, subset_size: int):
    if name == "metamathqa":
        ds = load_dataset("meta-math/MetaMathQA", split="train").shuffle(seed=SUBSET_SEED)
        return ds.select(range(min(subset_size, len(ds)))), "query", "response"
    if name == "magicoder":
        ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train").shuffle(seed=SUBSET_SEED)
        return ds.select(range(min(subset_size, len(ds)))), "instruction", "response"
    if name.startswith("glue:"):
        task = name.split(":", 1)[1]
        ds = load_dataset("glue", task, split="train").shuffle(seed=SUBSET_SEED)
        return ds.select(range(min(subset_size, len(ds)))), "sentence", "label"
    raise ValueError(f"unknown train dataset: {name}")


class CausalLMDataset(Dataset):
    """instruction -> response SFT packing with prompt-masked labels."""

    def __init__(self, hf_ds, prompt_col: str, answer_col: str, tokenizer, seq_len: int):
        self.ds = hf_ds
        self.prompt_col, self.answer_col = prompt_col, answer_col
        self.tok, self.seq_len = tokenizer, seq_len

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> dict:
        row = self.ds[i]
        if self.answer_col == "label":  # glue-style classification as verbalized SFT
            names = ["negative", "positive"]
            prompt = f"Sentence: {row[self.prompt_col]}\nSentiment:"
            answer = f" {names[row[self.answer_col]]}"
        else:
            prompt = f"Question: {row[self.prompt_col]}\nAnswer:"
            answer = f" {row[self.answer_col]}"
        p_ids = self.tok(prompt, add_special_tokens=True)["input_ids"]
        a_ids = self.tok(answer, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]
        ids = (p_ids + a_ids)[: self.seq_len]
        labels = ([-100] * len(p_ids) + a_ids)[: self.seq_len]
        pad = self.seq_len - len(ids)
        attn = [1] * len(ids) + [0] * pad
        ids = ids + [self.tok.pad_token_id or 0] * pad
        labels = labels + [-100] * pad
        return {"input_ids": ids, "attention_mask": attn, "labels": labels}


def _fixed_len_collate(feats: list[dict]) -> dict[str, torch.Tensor]:
    """All sequences are exactly seq_len -> plain stack; guarantees tensors (default_collate
    returns lists for variable-length batches, which breaks batch.to(device))."""
    keys = feats[0].keys()
    return {k: torch.tensor([f[k] for f in feats], dtype=torch.long) for k in keys}


def make_train_loader(cfg, tokenizer) -> DataLoader:
    ds, p_col, a_col = _load_raw(cfg.data.name, cfg.data.subset_size)
    wrapped = CausalLMDataset(ds, p_col, a_col, tokenizer, cfg.data.seq_len)
    return DataLoader(wrapped, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                      num_workers=2, collate_fn=_fixed_len_collate)


def make_gora_probe_loader(cfg, tokenizer, n_batches: int):
    """Small deterministic loader for GoRA's N-batch gradient pre-pass (uses head of the fixed subset)."""
    ds, p_col, a_col = _load_raw(cfg.data.name, cfg.data.subset_size)
    head = ds.select(range(min(n_batches * cfg.batch_size, len(ds))))
    wrapped = CausalLMDataset(head, p_col, a_col, tokenizer, cfg.data.seq_len)
    return DataLoader(wrapped, batch_size=cfg.batch_size, shuffle=False, drop_last=True,
                      collate_fn=_fixed_len_collate)


def load_eval_prompts(name: str, size: int):
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=SUBSET_SEED)
        rows = [dict(q=r["question"], gold=r["answer"].split("####")[-1].strip().replace(",", "")) for r in ds.select(range(min(size, len(ds))))]
        return rows
    if name.startswith("glue:"):
        task = name.split(":", 1)[1]
        split = "validation_matched" if task == "mnli" else "validation"
        ds = load_dataset("glue", task, split=split).shuffle(seed=SUBSET_SEED)
        rows = [dict(q=r["sentence"], gold=int(r["label"])) for r in ds.select(range(min(size, len(ds))))]
        return rows
    raise ValueError(f"unknown eval dataset: {name}")
