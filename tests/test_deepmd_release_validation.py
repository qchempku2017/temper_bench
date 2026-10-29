"""CPU coverage of validation data and cleanup; backend calls are mocked."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ase.calculators.emt import EMT


@pytest.mark.parametrize("molecular", [False, True])
def test_release_validation_inputs_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, molecular: bool,
) -> None:
    """Check emitted labels and conditioning without invoking DeepMD."""
    path = Path(__file__).parents[1] / "compatibility"
    spec = importlib.util.spec_from_file_location(
        "release_validation", path / "validate_deepmd_release.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"mock checkpoint")
    config = {"model": {"type_map": ["H", "Cu"]},
              "optimizer": {"type": "HybridMuon", "weight_decay": 0.001}}
    source = tmp_path / "source.json"
    source.write_text(json.dumps(config))
    output = tmp_path / "result"
    monkeypatch.setenv("SLURM_JOB_ID", "mock-job")
    monkeypatch.setattr(sys, "argv", [
        str(path), "dpa4", str(checkpoint), str(source), str(output),
    ] + (["--molecular"] if molecular else []))
    monkeypatch.setattr(module, "version", lambda name: "3.2.0")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True,
                             device_count=lambda: 1,
                             get_device_name=lambda: "mock GPU"),
        __version__="mock", version=SimpleNamespace(cuda="mock"),
    ))
    monkeypatch.setitem(sys.modules, "deepmd.calculator", SimpleNamespace(
        DP=lambda **kwargs: EMT(),
    ))

    def run(command: list[str], **kwargs: object) -> None:
        work = Path(str(kwargs["cwd"]))
        if "train" in command:
            generated = json.loads((work / "input.json").read_text())
            assert generated["model"] == config["model"]
            assert generated["optimizer"] == config["optimizer"]
            arrays = work / "train" / "set.000"
            count = 2 if molecular else 4
            assert np.load(arrays / "force.npy").shape == (2, count * 3)
            assert np.load(arrays / "virial.npy").shape == (2, 9)
            assert np.isfinite(np.load(arrays / "energy.npy")).all()
            if molecular:
                np.testing.assert_array_equal(
                    np.load(arrays / "charge_spin.npy"), [[0, 1], [0, 1]]
                )
            else:
                assert not (arrays / "charge_spin.npy").exists()
            shutil.copyfile(work / "input.json", work / "adapted.json")
        else:
            (work / "evaluation.pt2").write_bytes(b"mock export")

    monkeypatch.setattr(module.subprocess, "run", run)
    module.main()
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "passed"
    assert result["export_cleanup"] == "passed"
    assert not (output / "work").exists()
    assert checkpoint.read_bytes() == b"mock checkpoint"
    assert json.loads(source.read_text()) == config


@pytest.mark.parametrize("previous_molecular", [False, True])
def test_resume_rejects_changed_molecular_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    previous_molecular: bool,
) -> None:
    """Do not reuse training with a different toy-data conditioning mode."""
    path = Path(__file__).parents[1] / "compatibility"
    spec = importlib.util.spec_from_file_location(
        "resume_validation", path / "validate_deepmd_release.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    configuration = tmp_path / "model.json"
    configuration.write_text("{}")
    output = tmp_path / "output"
    output.mkdir()
    previous = {
        "train": "passed", "molecular": previous_molecular,
        "checkpoint_sha256": hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest(),
        "configuration_sha256": hashlib.sha256(
            configuration.read_bytes()
        ).hexdigest(),
    }
    (output / "result.json").write_text(json.dumps(previous))
    monkeypatch.setenv("SLURM_JOB_ID", "mock-job")
    monkeypatch.setattr(sys, "argv", [
        "validate.py", "dpa4", str(checkpoint), str(configuration),
        str(output), "--resume-export",
    ] + ([] if previous_molecular else ["--molecular"]))
    with pytest.raises(ValueError, match="molecular mode"):
        module.main()
    assert json.loads((output / "result.json").read_text()) == previous
    assert not (output / "work").exists()
