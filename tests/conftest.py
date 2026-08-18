"""Shared deterministic fixtures for schema and grouping tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


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
