"""Checkpoint/resume: adapter + optimizer state + step, for Kaggle session splits."""
from __future__ import annotations

import os
from typing import Optional

import torch


def save_ckpt(ckpt_dir: str, exp_id: str, model, optimizer, step: int) -> str:
    path = os.path.join(ckpt_dir, exp_id)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(os.path.join(path, "adapter"))  # peft adapter (small)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        os.path.join(path, "state.pt"),
    )
    return path


def load_ckpt(ckpt_dir: str, exp_id: str, model, optimizer) -> Optional[int]:
    """Returns resumed step or None. Caller re-creates dataloader order if needed (documented)."""
    path = os.path.join(ckpt_dir, exp_id)
    state_path = os.path.join(path, "state.pt")
    if not os.path.isfile(state_path):
        return None
    adapter_path = os.path.join(path, "adapter")
    if os.path.isdir(adapter_path):
        model.load_adapter(adapter_path, adapter_name="default", is_trainable=True)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    return int(state["step"])
