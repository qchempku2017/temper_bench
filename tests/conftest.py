"""Shared deterministic fixtures for schema and grouping tests."""
from __future__ import annotations

import sys
import json
from pathlib import Path
from uuid import UUID

import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from temper.mlff.spec_builders import (
    DPA4CSpecBuilder,
    DPA4SpecBuilder,
    MACESpecBuilder,
    MatterSimSpecBuilder,
    NEP89SpecBuilder,
    SevenNetSpecBuilder,
)
from temper.schemas.train_unit import TrainingUnit


NEP_MODEL = b"""nep4_zbl 2 H He
zbl 1 2
cutoff 6 5 20 20
n_max 4 4
basis_size 8 8
l_max 4 2 1
ANN 80 0
0
"""


def make_frame(symbols: str, energy: float, tag: str, *, stress: bool = True) -> Atoms:
    """Create a small extxyz-compatible frame with deterministic properties."""
    atoms = Atoms(symbols)
    atoms.positions = [(index, 0, 0) for index in range(len(atoms))]
    results = {
        "energy": energy,
        "forces": [[0.0, 0.0, 0.0] for _ in atoms],
    }
    if stress:
        results["stress"] = [0.0] * 6
    atoms.calc = SinglePointCalculator(atoms, **results)
    atoms.info["dataset_tag"] = tag
    return atoms


@pytest.fixture
def extxyz_domain(tmp_path: Path) -> Path:
    """A two-file domain with two deterministic frames per source file."""
    domain = tmp_path / "demo_domain"
    domain.mkdir()
    write(domain / "alpha_t_300_run.extxyz", [
        make_frame("H2", -1.0, "shared"),
        make_frame("H2", -1.1, "shared"),
    ], format="extxyz")
    write(domain / "beta_t_600_run.extxyz", [
        make_frame("He", -0.2, "shared"),
        make_frame("He", -0.3, "shared"),
    ], format="extxyz")
    return domain


@pytest.fixture
def metadata_payload() -> dict:
    """Required metadata paired with the files created by extxyz_domain."""
    return {
        "info": [
            {"filename": "alpha_t_300_run.extxyz", "name": "alpha", "source": "unit-test", "domain": "demo_domain", "system_type": ["molecule"]},
            {"filename": "beta_t_600_run.extxyz", "name": "beta", "source": "unit-test", "domain": "demo_domain", "system_type": ["atom"]},
        ]
    }


@pytest.fixture
def mlff_dataset_root(tmp_path: Path) -> Path:
    """Small exported-data tree used by local MLFF bundle tests."""
    root = tmp_path / "split_results"
    domain = root / "demo_domain"
    domain.mkdir(parents=True)

    def periodic(frame: Atoms) -> Atoms:
        frame.set_cell([10.0, 10.0, 10.0])
        frame.set_pbc(True)
        return frame

    write(
        domain / "train.extxyz",
        [periodic(make_frame("H2", -1.0, "train"))],
        format="extxyz",
    )
    write(
        domain / "validation.extxyz",
        [periodic(make_frame("H", -0.4, "validation"))],
        format="extxyz",
    )
    write(
        domain / "test_own.extxyz",
        [
            periodic(make_frame("H", -0.3, "test-own-0")),
            periodic(make_frame("H2", -0.9, "test-own-1")),
        ],
        format="extxyz",
    )
    write(
        domain / "test_cross.extxyz",
        [periodic(make_frame("He2", -0.2, "test-cross"))],
        format="extxyz",
    )
    return root


@pytest.fixture
def finetune_training_unit(mlff_dataset_root: Path) -> TrainingUnit:
    """One fine-tuning unit with validation and a cross-group test."""
    return TrainingUnit(
        domain="demo_domain",
        grouping_strategy="as_specified",
        group_name="hydrogen",
        method="random",
        repeat_id=0,
        train_n_frames=1,
        val_n_frames=1,
        test_n_frames=3,
        train_n_atoms=2,
        val_n_atoms=1,
        test_n_atoms=5,
        split_id=UUID("0c112fa7-ec27-5ee2-829d-ef1cb92e8238"),
        train_set="train.extxyz",
        val_set="validation.extxyz",
        test_sets=("test_own.extxyz", "test_cross.extxyz"),
        root_path=mlff_dataset_root,
    )


@pytest.fixture
def zeroshot_training_unit(mlff_dataset_root: Path) -> TrainingUnit:
    """The test-only counterpart of ``finetune_training_unit``."""
    return TrainingUnit(
        domain="demo_domain",
        grouping_strategy="as_specified",
        group_name="hydrogen",
        method="random",
        repeat_id=0,
        train_n_frames=0,
        val_n_frames=0,
        test_n_frames=3,
        train_n_atoms=0,
        val_n_atoms=0,
        test_n_atoms=5,
        split_id=UUID("0c112fa7-ec27-5ee2-829d-ef1cb92e8238"),
        train_set=None,
        val_set=None,
        test_sets=("test_own.extxyz", "test_cross.extxyz"),
        root_path=mlff_dataset_root,
    )


@pytest.fixture
def mlff_spec_factory(tmp_path: Path):
    """Build six-family specs without importing third-party MLFF packages."""
    artifact_root = tmp_path / "model_artifacts"
    artifact_root.mkdir(exist_ok=True)

    def artifact(family: str, name: str, content: bytes) -> Path:
        path = artifact_root / f"{family}-{name}"
        path.write_bytes(content)
        return path

    def factory(
        family: str,
        *,
        with_training: bool = True,
        training_parameters: dict | None = None,
        testing_parameters: dict | None = None,
    ):
        if family in {"dpa4", "dpa4c"}:
            config_path = artifact_root / f"{family}-config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model": {"type_map": ["H", "He"]},
                        "training": {},
                        "loss": {},
                    }
                ),
                encoding="utf-8",
            )
            builder_type = (
                DPA4SpecBuilder
                if family == "dpa4"
                else DPA4CSpecBuilder
            )
            paths = {
                "pretrained_model_path": artifact(
                    family, "model.pt", b"deepmd-checkpoint"
                ),
                "pretrained_config_path": config_path,
            }
        elif family == "mattersim":
            builder_type = MatterSimSpecBuilder
            paths = {
                "pretrained_model_path": artifact(
                    family, "model.pth", b"mattersim-checkpoint"
                )
            }
        elif family == "mace":
            builder_type = MACESpecBuilder
            paths = {
                "pretrained_model_path": artifact(
                    family, "model.model", b"mace-checkpoint"
                )
            }
        elif family == "sevennet":
            builder_type = SevenNetSpecBuilder
            paths = {
                "pretrained_model_path": artifact(
                    family, "model.pth", b"sevennet-checkpoint"
                )
            }
        elif family == "nep89":
            builder_type = NEP89SpecBuilder
            paths = {
                "pretrained_model_path": artifact(
                    family, "model.txt", NEP_MODEL
                )
            }
        else:
            raise ValueError(f"Unsupported test MLFF family: {family}.")

        if with_training:
            training = {} if training_parameters is None else training_parameters
        else:
            training = None
        return builder_type(
            **paths,
            training_parameters=training,
            testing_parameters=testing_parameters,
        ).build()

    return factory
