from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_split_entrypoint_passes_deterministic_config_and_writes_json(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import src.temper.entrypoints.split as entrypoint
    captured = []
    monkeypatch.setattr(entrypoint, "partition_domain_groups", lambda domain, root_path: ["grouped"])
    monkeypatch.setattr(entrypoint, "split_grouped_domain", lambda grouped, config: captured.append(config) or [SimpleNamespace(model_dump_json=lambda indent: '{"ok": true}')])
    destination = tmp_path / "splits.json"
    assert entrypoint.main(["demo", "--root-path", str(tmp_path), "--output", str(destination), "--method", "random", "--split-repeats", "2", "--seed", "7"]) == 0
    assert captured[0].trainval_test_split_seeds == [7, 8]
    assert captured[0].train_val_split_seeds == [9, 10]
    assert destination.read_text(encoding="utf-8") == '[\n{"ok": true}\n]\n'
