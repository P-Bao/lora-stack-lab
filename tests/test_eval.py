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
