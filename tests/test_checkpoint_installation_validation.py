"""Check manifest selection without importing or executing GPU backends."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.mark.parametrize("failure", [None, "checkpoint", "configuration"])
def test_installation_selects_matching_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
) -> None:
    """Select a second same-family entry and reject unmatched inputs."""
    source = Path(__file__).parents[1] / "compatibility/check_installation.py"
    spec = importlib.util.spec_from_file_location("installation_check", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compatibility = tmp_path / "compatibility"
    compatibility.mkdir()
    manifest_directory = tmp_path / "docs/deployment"
    manifest_directory.mkdir(parents=True)
    checkpoint = tmp_path / "second.pt"
    checkpoint.write_bytes(b"second checkpoint")
    configuration = checkpoint.with_suffix(".json")
    configuration.write_text('{"model": {"type_map": ["H"]}}')
    expected = {
        "family": "dpa4",
        "variant": "second",
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "configuration_sha256": hashlib.sha256(
            configuration.read_bytes()
        ).hexdigest(),
    }
    manifest = [
        {**expected, "variant": "first", "sha256": "0" * 64}, expected,
    ]
    (manifest_directory / "pretrained-models.json").write_text(
        json.dumps(manifest)
    )
    (compatibility / "patched-source-sha256.json").write_text("{}")
    if failure == "checkpoint":
        checkpoint.write_bytes(b"unvalidated checkpoint")
    elif failure == "configuration":
        configuration.write_text("{}")
    output = tmp_path / "report.json"
    monkeypatch.setattr(module, "__file__", str(compatibility / "check.py"))
    monkeypatch.setattr(sys, "argv", [
        "check.py", "dpa4", str(checkpoint), str(output),
    ])
    versions = {
        "deepmd-kit": "3.2.0", "torch": "2.12.0",
        "vesin": "0.6.1", "vesin-torch": "0.6.1",
    }
    monkeypatch.setattr(module, "version", versions.__getitem__)
    monkeypatch.setattr(module.shutil, "which", lambda name: "/mock/bin/dp")
    calls: list[list[str]] = []

    def run(command: list[str], *, check: bool) -> None:
        assert check
        calls.append(command)

    monkeypatch.setattr(module.subprocess, "run", run)
    if failure is not None:
        with pytest.raises(ValueError, match="[Cc]heckpoint"):
            module.main()
        assert not output.exists()
        assert calls == []
    else:
        module.main()
        report = json.loads(output.read_text())
        assert report["checkpoint"] == expected
        assert report["gpu_functional_test"] == "not performed"
        assert calls == [[sys.executable, "-m", "pip", "check"]]
