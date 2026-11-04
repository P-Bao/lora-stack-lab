"""Unit tests for eval extraction + reeval-preference aggregation (no GPU needed)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from lorastack.eval.evaluators import _last_number  # noqa: E402


def test_last_number_basic():
    assert _last_number("The answer is: 72") == "72"
    assert _last_number("So x = 1,024 in total.") == "1024"
    assert _last_number("temp is -3.5 degrees") == "-3.5"
    assert _last_number("no digits here") is None


def test_last_number_gsm8k_gold():
    gold = "Natalia sold 48 clips in May.\nNatalia sold 24 clips in June.\n#### 72"
    assert _last_number(gold) == "72"


def test_agg_prefers_reeval(tmp_path, monkeypatch, capsys):
    import aggregate

    rec = {"status": "done", "id": "R1", "config": {"components": ["L"]},
           "metrics": {"gsm8k_acc": 0.0, "n_eval": 2}}
    reeval = {**rec, "reeval": True, "metrics": {"gsm8k_acc": 0.5, "n_eval": 2}}
    (tmp_path / "R1.json").write_text(json.dumps(rec), encoding="utf-8")
    (tmp_path / "R1.reeval.json").write_text(json.dumps(reeval), encoding="utf-8")
    (tmp_path / "R1.reeval.jsonl").write_text("", encoding="utf-8")  # *.json glob must ignore
    monkeypatch.setattr(sys, "argv", ["aggregate.py", "--results", str(tmp_path)])
    assert aggregate.main() == 0
    out = capsys.readouterr().out
    assert "50.00" in out            # reeval score wins
    assert "   1    0.00" not in out  # old broken score must not appear as mean


def test_reeval_inherits_components(tmp_path):
    """Direct run json carries true components; fallback when absent/corrupt (stage2 gate table)."""
    import reeval

    (tmp_path / "stage2-G-lr2e-4-r32-s1.json").write_text(
        json.dumps({"status": "done", "config": {"components": ["G"]}}), encoding="utf-8")
    assert reeval.inherit_components(tmp_path, "stage2-G-lr2e-4-r32-s1", ["L"]) == ["G"]
    assert reeval.inherit_components(tmp_path, "no-direct-json", ["L"]) == ["L"]      # fallback
    (tmp_path / "bad.json").write_text("{oops", encoding="utf-8")                     # corrupt json
    assert reeval.inherit_components(tmp_path, "bad", ["L"]) == ["L"]
    (tmp_path / "empty.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    assert reeval.inherit_components(tmp_path, "empty", ["L"]) == ["L"]


def test_reeval_needs_merge_only_plain_lora():
    """stage2 dataset: GRALORA adapters must not hit merge_and_unload (never run on GPU);
    pissa re-derives W_res at load -> unmerged as well. Plain LoRA still merges."""
    import reeval

    assert reeval.needs_merge({"peft_type": "LORA"}) is True
    assert reeval.needs_merge({"peft_type": "LORA", "init_lora_weights": "pissa"}) is False
    assert reeval.needs_merge({"peft_type": "GRALORA", "r": 32}) is False
