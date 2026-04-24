from __future__ import annotations

from pathlib import Path


class _SplitDict(dict):
    pass


def test_aime_split_sizes(monkeypatch):
    from apa.benchmarks import aime

    train_rows = [{"problem": f"p{i}", "solution": "s", "answer": i % 10} for i in range(40)]
    test_rows = [{"problem": f"tp{i}", "answer": i % 10} for i in range(10)]

    def fake_load_dataset(name, *args, **kwargs):
        if name == "AI-MO/aimo-validation-aime":
            return _SplitDict(train=train_rows)
        if name == "MathArena/aime_2025":
            return _SplitDict(train=test_rows)
        raise AssertionError(name)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    runtime = type("R", (), {"dataset_mode": "full", "smoke": False})()
    splits = aime._load_splits(runtime)
    assert len(splits.train) == 20
    assert len(splits.val) == 20
    assert len(splits.test) == 50


def test_livebench_split_sizes(monkeypatch):
    from apa.benchmarks import livebench_math

    rows = [{"turns": [f"q{i}"], "ground_truth": str(i), "task": "aime_2024", "question_id": i} for i in range(30)]

    def fake_load_dataset(name, *args, **kwargs):
        assert name == "livebench/math"
        return _SplitDict(test=rows)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    runtime = type("R", (), {"dataset_mode": "full", "smoke": False})()
    splits = livebench_math._load_splits(runtime)
    assert len(splits.train) == 9
    assert len(splits.val) == 10
    assert len(splits.test) == 11


def test_ifbench_split_sizes(monkeypatch):
    from apa.benchmarks import ifbench

    train_rows = [{"prompt": f"p{i}", "instruction_id_list": [], "kwargs": []} for i in range(700)]
    test_rows = [{"prompt": f"t{i}", "instruction_id_list": [], "kwargs": []} for i in range(50)]

    monkeypatch.setattr(ifbench, "_ensure_ifbench_files", lambda: (Path("train.jsonl"), Path("test.jsonl")))

    def fake_load_jsonl(path):
        if path.name.startswith("train"):
            return train_rows
        return test_rows

    monkeypatch.setattr(ifbench, "_load_jsonl", fake_load_jsonl)

    runtime = type("R", (), {"dataset_mode": "full", "smoke": False})()
    splits = ifbench._load_splits(runtime)
    assert len(splits.train) == 300
    assert len(splits.val) == 300
    assert len(splits.test) == 50


def test_pupa_split_sizes(monkeypatch):
    from apa.benchmarks import pupa

    rows = [
        {"target_response": "t", "user_query": f"q{i}", "pii_units": ""}
        for i in range(500)
    ]

    def fake_load_dataset(name, subset, *args, **kwargs):
        assert name == "Columbia-NLP/PUPA"
        assert subset == "pupa_new"
        return _SplitDict(train=rows)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    runtime = type("R", (), {"dataset_mode": "full", "smoke": False})()
    splits = pupa._load_splits(runtime)
    assert len(splits.train) == 111
    assert len(splits.val) == 111
    assert len(splits.test) == 221


def test_hover_and_hotpot_split_sizes(monkeypatch):
    from apa.benchmarks import hotpotqa, hover

    hover_rows = [
        {
            "claim": f"c{i}",
            "supporting_facts": [{"key": "DocA"}, {"key": "DocB"}, {"key": "DocC"}],
            "label": "SUPPORTED",
        }
        for i in range(900)
    ]

    hotpot_rows = [
        {
            "question": f"q{i}",
            "answer": "a",
            "context": {"title": ["DocA"], "sentences": [["sentence"]]},
            "supporting_facts": {"title": ["DocA"], "sent_id": [0]},
        }
        for i in range(900)
    ]

    def fake_load_dataset(name, *args, **kwargs):
        if name == "hover":
            return _SplitDict(train=hover_rows)
        if name == "hotpot_qa":
            return _SplitDict(train=hotpot_rows)
        raise AssertionError(name)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    runtime = type("R", (), {"dataset_mode": "full", "smoke": False})()

    hover_splits = hover._load_splits(runtime)
    hotpot_splits = hotpotqa._load_splits(runtime)

    assert len(hover_splits.train) == 150
    assert len(hover_splits.val) == 300
    assert len(hover_splits.test) == 300

    assert len(hotpot_splits.train) == 150
    assert len(hotpot_splits.val) == 300
    assert len(hotpot_splits.test) == 300
