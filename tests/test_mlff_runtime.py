"""Generic ASE test-runtime behavior without real MLFF installations."""

from __future__ import annotations

import json
import sys
from importlib import resources
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write

from temper.mlff.runtime import run_test, train_nep89


class FakeCalculator(Calculator):
    """Deterministic energy/force/stress calculator for runtime tests."""

    implemented_properties = ["energy", "forces", "stress"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces", "stress"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        assert atoms is not None
        atom_count = len(atoms)
        self.results = {
            "energy": float(atom_count),
            "forces": np.full((atom_count, 3), float(atom_count)),
            # ASE's six-vector order is xx, yy, zz, yz, xz, xy. The generic
            # runner requests voigt=False and persists the resulting tensor.
            "stress": np.arange(6, dtype=float),
        }


class NoStressCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        assert atoms is not None
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(atoms), 3)),
        }


def _write_runtime_config(root: Path, *, stress: bool = True) -> Path:
    datasets = root / "datasets"
    datasets.mkdir(parents=True)
    first = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    second = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]],
    )
    third = Atoms("He", positions=[[0.0, 0.0, 0.0]])
    write(datasets / "first.extxyz", [first, second], format="extxyz")
    write(datasets / "second.extxyz", [third], format="extxyz")

    properties = ["energy", "forces"]
    if stress:
        properties.append("stress")
    config = {
        "schema_version": 2,
        "calculator": {
            "identifier": "fake",
            "parameters": {},
        },
        "model": "models/fake-model",
        "test_datasets": [
            {
                "id": "test_000",
                "path": "datasets/first.extxyz",
                "source_domain": "domain",
                "source_filename": "first.extxyz",
                "properties": properties,
                "output": "outputs/test_000.npz",
                "metadata_output": "outputs/test_000.json",
            },
            {
                "id": "test_001",
                "path": "datasets/second.extxyz",
                "source_domain": "domain",
                "source_filename": "second.extxyz",
                "properties": properties,
                "output": "outputs/test_001.npz",
                "metadata_output": "outputs/test_001.json",
            },
        ],
        "summary_output": "outputs/summary.json",
        "package_requirements": [],
    }
    path = root / "test_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _install_fake_adapter(monkeypatch, calculator: Calculator) -> None:
    module = ModuleType("calculator")
    module.build_calculator = lambda _config: calculator
    monkeypatch.setitem(sys.modules, "calculator", module)


def test_generic_runner_preserves_frames_and_writes_ragged_arrays(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _write_runtime_config(tmp_path)
    _install_fake_adapter(monkeypatch, FakeCalculator())
    run_test.run(config)

    with np.load(tmp_path / "outputs" / "test_000.npz", allow_pickle=False) as data:
        assert data.files == [
            "energies",
            "forces",
            "atom_offsets",
            "frame_indices",
            "stresses",
        ]
        np.testing.assert_array_equal(data["energies"], [1.0, 2.0])
        np.testing.assert_array_equal(data["atom_offsets"], [0, 1, 3])
        np.testing.assert_array_equal(data["frame_indices"], [0, 1])
        assert data["forces"].shape == (3, 3)
        np.testing.assert_array_equal(data["forces"][0], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(data["forces"][1:], np.full((2, 3), 2.0))
        assert data["stresses"].shape == (2, 3, 3)
        np.testing.assert_array_equal(
            data["stresses"][0],
            [[0.0, 5.0, 4.0], [5.0, 1.0, 3.0], [4.0, 3.0, 2.0]],
        )
        assert all(data[name].dtype != object for name in data.files)

    with np.load(tmp_path / "outputs" / "test_001.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["energies"], [1.0])
        np.testing.assert_array_equal(data["atom_offsets"], [0, 1])

    first_metadata = json.loads(
        (tmp_path / "outputs" / "test_000.json").read_text(encoding="utf-8")
    )
    assert first_metadata["dataset_id"] == "test_000"
    assert first_metadata["source_filename"] == "first.extxyz"
    assert first_metadata["number_of_frames"] == 2
    assert first_metadata["total_atoms"] == 3
    assert first_metadata["units"] == {
        "energy": "eV",
        "forces": "eV/Angstrom",
        "stress": "eV/Angstrom^3",
    }

    summary = json.loads(
        (tmp_path / "outputs" / "summary.json").read_text(encoding="utf-8")
    )
    assert [item["dataset_id"] for item in summary["datasets"]] == [
        "test_000",
        "test_001",
    ]


def test_generic_runner_omits_stress_when_not_requested(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _write_runtime_config(tmp_path, stress=False)
    _install_fake_adapter(monkeypatch, FakeCalculator())
    run_test.run(config)

    with np.load(tmp_path / "outputs" / "test_000.npz", allow_pickle=False) as data:
        assert "stresses" not in data.files
    metadata = json.loads(
        (tmp_path / "outputs" / "test_000.json").read_text(encoding="utf-8")
    )
    assert metadata["units"]["stress"] is None


def test_generic_runner_reports_unsupported_stress_with_dataset_and_frame(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _write_runtime_config(tmp_path, stress=True)
    _install_fake_adapter(monkeypatch, NoStressCalculator())
    with pytest.raises(RuntimeError, match="test_000 frame 0"):
        run_test.run(config)


def test_generic_runner_rejects_dataset_escape(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path, stress=False)
    outside = tmp_path.parent / "outside.extxyz"
    write(outside, Atoms("H"), format="extxyz")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["test_datasets"][0]["path"] = "../outside.extxyz"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _install_fake_adapter(monkeypatch, FakeCalculator())
    with pytest.raises(ValueError, match="escapes bundle root"):
        run_test.run(config_path)


def test_generic_runner_has_no_mlff_dispatch_or_temper_dependency() -> None:
    source = (
        resources.files("temper.mlff.runtime")
        .joinpath("run_test.py")
        .read_text(encoding="utf-8")
    )
    assert "import temper" not in source
    assert "if mlff" not in source.lower()
    for identifier in ("dpa4", "mattersim", "mace", "sevennet", "nep89", "orb"):
        assert identifier not in source.lower()


@pytest.mark.parametrize(
    "resource",
    (
        "calculators/deepmd.py",
        "calculators/mattersim.py",
        "calculators/mace.py",
        "calculators/sevennet.py",
        "calculators/nep89.py",
    ),
)
def test_every_calculator_resource_implements_static_contract(
    resource: str,
) -> None:
    source = (
        resources.files("temper.mlff.runtime")
        .joinpath(resource)
        .read_text(encoding="utf-8")
    )
    assert "def build_calculator(config):" in source
    assert "import temper" not in source


def test_orb_runtime_resources_are_absent() -> None:
    runtime = resources.files("temper.mlff.runtime")
    assert not runtime.joinpath("calculators/orb.py").is_file()
    assert not runtime.joinpath("data_preparation/orb.py").is_file()


def test_torchnep_wrapper_starts_one_fresh_finetune(monkeypatch) -> None:
    captured = {}
    module = ModuleType("torchnep")

    def fake_train_nep(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    module.train_nep = fake_train_nep
    monkeypatch.setitem(sys.modules, "torchnep", module)

    train_nep89.run("nep.in", "train.xyz", "valid.xyz", "nep.txt", "out")

    assert captured["args"] == ("nep.in", "train.xyz")
    assert captured["kwargs"] == {
        "output_dir": "out",
        "device": "cuda",
        "finetune_from": "nep.txt",
        "restart": False,
        "recompute_q_scaler": False,
        "slim_types": True,
        "run_seed": 42,
        "valid_file": "valid.xyz",
    }
