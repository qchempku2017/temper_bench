"""Regression tests for metadata scans without ASE structure construction."""

import numpy as np
import pytest
from ase.io import read, write

from conftest import make_frame
from temper.schemas.info import InfoEntry
from temper.schemas.train_unit import TrainingUnit
from temper.schemas.utils import check_atoms_have_other_properties
from temper.utils.extxyz import check_extxyz_properties, iter_extxyz_metadata


def test_metadata_matches_ase_for_mixed_systems(tmp_path):
    path = tmp_path / "mixed.extxyz"
    frames = [
        make_frame("OH2", -1.0, "one"),
        make_frame("He", -2.0, "two"),
        make_frame("H2O", -3.0, "three"),
    ]
    for frame in frames:
        frame.info["note"] = "quoted text with spaces"
        frame.calc.results["charges"] = np.zeros(len(frame))
    write(path, frames, format="extxyz")
    reference = read(path, index=":")
    with pytest.warns(UserWarning, match="Missing optional"):
        entry = InfoEntry.from_extxyz(path, source="test", system_type=["molecule"])
    assert entry.formulas == ["H2O", "He"]
    assert entry.num_frames_per_system == [2, 1]
    assert entry.num_atoms_per_system == [3, 1]
    assert entry.has_stress
    assert entry.has_other_properties == check_atoms_have_other_properties(reference)


@pytest.mark.parametrize(
    ("declaration", "row", "symbols"),
    [
        ("pos:R:3:species:S:1:forces:R:3", "x y z He a b c", ("He",)),
        ("forces:R:3:Z:I:1:pos:R:3", "a b c 2 x y z", ("He",)),
    ],
)
def test_species_scan_skips_numerical_columns(tmp_path, declaration, row, symbols):
    path = tmp_path / "columns.extxyz"
    path.write_text(
        f'1\nProperties={declaration} energy=-1 pbc="T T F" note="two words"\n{row}\n'
    )
    frame = next(iter_extxyz_metadata(path, read_symbols=True))
    assert frame.symbols == symbols
    assert frame.info["note"] == "two words"
    assert not frame.fully_periodic
    assert not check_extxyz_properties(frame, source=str(path))
    assert next(iter_extxyz_metadata(path)).symbols == ()


def test_virial_is_detected_without_claiming_ase_stress(tmp_path):
    path = tmp_path / "virial.extxyz"
    path.write_text(
        '1\nProperties=species:S:1:pos:R:3:forces:R:3 energy=0 '
        'Lattice="2 0 0 0 2 0 0 0 2" virial="1 0 0 0 1 0 0 0 1"\n'
        'H 0 0 0 0 0 0\n'
    )
    frame = next(iter_extxyz_metadata(path))
    assert "virial" in frame.info
    assert frame.fully_periodic
    assert not check_extxyz_properties(frame, source=str(path))


@pytest.mark.parametrize("missing", ["energy", "forces"])
def test_every_frame_requires_energy_and_forces(tmp_path, missing):
    path = tmp_path / "missing.extxyz"
    frames = [make_frame("H", -1.0, "first"), make_frame("H", -1.0, "second")]
    del frames[1].calc.results[missing]
    write(path, frames, format="extxyz")
    with pytest.raises(ValueError, match=f"frame 1 is missing {missing}"):
        TrainingUnit._dataset_has_stress(path)


def test_mixed_stress_and_empty_datasets_are_rejected(tmp_path):
    path = tmp_path / "mixed.extxyz"
    write(path, [make_frame("H", -1, "one"), make_frame("H", -1, "two", stress=False)])
    with pytest.raises(ValueError, match="mixes frames"):
        TrainingUnit._dataset_has_stress(path)
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        TrainingUnit._dataset_has_stress(path)
