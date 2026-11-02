"""Batched generation path of eval_gsm8k: stub tokenizer/model, no downloads, no GPU."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from lorastack.config import ExpConfig  # noqa: E402
from lorastack.eval import evaluators  # noqa: E402


class StubEnc:
    def __init__(self, rows):
        self.input_ids = torch.tensor(rows)

    def keys(self):  # mapping protocol: evaluator unpacks with **enc
        return ["input_ids"]

    def to(self, device):
        return self

    def __getitem__(self, key):
        return self.input_ids


class StubTok:
    padding_side = "right"
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, prompts, return_tensors="pt", padding=False, add_special_tokens=True):
        rows = [[7 + i] * (i + 3) for i in range(len(prompts))]  # row i has i+3 prompt tokens
        width = max(len(r) for r in rows)
        if padding and self.padding_side == "left":
            rows = [[0] * (width - len(r)) + r for r in rows]
        return StubEnc(rows)

    def batch_decode(self, batch, skip_special_tokens=True):
        # batch rows are the 3 new tokens [10+r, 11, 12]; id 10+r carries the answer
        return [f"The answer is: {20 + row[0] - 10}." for row in batch]


class StubModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kw):
        self.calls.append(kw)
        enc = kw["input_ids"]
        new = torch.tensor([[10 + r, 11, 12] for r in range(enc.shape[0])])
        return torch.cat([enc, new], dim=1)


def test_eval_gsm8k_batched(monkeypatch):
    rows = [{"q": "Q1", "gold": "20"}, {"q": "Q2", "gold": "99"}]
    monkeypatch.setattr(evaluators, "load_eval_prompts", lambda name, size: rows[:size])
    tok, model = StubTok(), StubModel()
    cfg = ExpConfig()
    cfg.data.eval_size, cfg.data.eval_batch_size = 2, 16

    metrics = evaluators.eval_gsm8k(model, tok, cfg, "cpu")

    assert len(model.calls) == 1                    # 2 prompts, batch 16 -> one generate call
    ids = model.calls[0]["input_ids"]
    assert ids.shape == (2, 4)                      # left-padded to common width + uniform
    assert ids[0][0].item() == 0                    # shorter prompt is padded on the LEFT
    assert metrics == {"gsm8k_acc": 0.5, "n_eval": 2}  # row1 pred 20 == gold 20; row2 pred 21 != 99
