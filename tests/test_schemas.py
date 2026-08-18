"""Focused tests for persisted schemas and metadata discovery."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from pydantic import ValidationError

from src.temper.schemas.group import GroupedDomain
from src.temper.schemas.info import InfoEntry, load_info_entries_from_datadir
from src.temper.schemas.split import (
    SplitGroup,
    TrainValSplitTrajectory,
)
from src.temper.schemas.entropy import EntropyProfilePoint, EntropyProfile
from src.temper.schemas.frame_refrence import FrameReference
from src.temper.schemas.train_unit import TrainingUnit
from src.temper.schemas.utils import (
    check_atoms_has_stress,
    check_atoms_have_other_properties,
    validate_relative_extxyz_path,
)


def required_info(filename: str = "sample.extxyz") -> dict:
    return {
        "name": "sample",
        "source": "unit-test",
        "domain": "domain",
        "filename": filename,
        "system_type": ["molecule"],
    }


def references() -> list[FrameReference]:
    return [FrameReference(domain="domain", filename="sample.extxyz", frame_index=i) for i in range(4)]


def test_json_models_round_trip_without_data_loss(tmp_path: Path) -> None:
    info = InfoEntry(**required_info(), description="deterministic")
    refs = references()
    trajectory = TrainValSplitTrajectory(
        method="random", seed=7, requested_train_sizes=[1, 2],
        selected_frames=refs[:2], additional_trainval_frames=[refs[2]],
        entropy_profile=EntropyProfile(points=[EntropyProfilePoint(training_size=1, cumulative_entropy=-1.0, information_gain=-0.2)]),
    )
    split = SplitGroup(
        domain="domain", grouping_strategy="all", group_name="all", test_set=[refs[3]],
        extra_tested_groups=[], test_ratio=0.25, trainval_test_split_seed=9,
        train_val_split_trajectory=trajectory,
    )
    grouped = GroupedDomain(domain="domain", info_entries=[info], grouping_strategy="all", groups={"all": ["sample.extxyz"]})

    for model in (info, refs[0], trajectory, split, grouped):
        path = tmp_path / f"{type(model).__name__}.json"
        model.save_json(path)
        assert type(model).load_json(path) == model


def test_atom_property_helpers_validate_and_report_common_properties() -> None:
    valid = Atoms("H")
    valid.calc = SinglePointCalculator(valid, energy=0.0, forces=[[0, 0, 0]], stress=[0] * 6, dipole=[1, 0, 0])
    valid.info["label"] = "common"
    second = valid.copy()
    second.calc = SinglePointCalculator(second, energy=1.0, forces=[[0, 0, 0]], stress=[0] * 6, dipole=[0, 0, 0])
    second.info["label"] = "common"
    second.info["only_second"] = True

    assert check_atoms_has_stress([valid, second]) is True
    assert check_atoms_have_other_properties([valid, second]) == ["dipole", "label"]
    assert check_atoms_have_other_properties([second]) == ["dipole", "label", "only_second"]

    missing_stress = Atoms("H")
    missing_stress.calc = SinglePointCalculator(missing_stress, energy=0.0, forces=[[0, 0, 0]])
    with pytest.warns(UserWarning, match="Stress information is missing"):
        assert check_atoms_has_stress(missing_stress) is False

    missing_energy = Atoms("H")
    missing_energy.calc = SinglePointCalculator(missing_energy, forces=[[0, 0, 0]])
    with pytest.raises(ValueError, match="missing energy"):
        check_atoms_has_stress(missing_energy)


@pytest.mark.parametrize("path", ["file.extxyz", "subdir/file.extxyz"])
def test_relative_extxyz_paths_are_accepted(path: str) -> None:
    assert validate_relative_extxyz_path(path) == path


@pytest.mark.parametrize("path", ["../file.extxyz", "/file.extxyz", "file.xyz"])
def test_relative_extxyz_paths_reject_unsafe_or_wrong_suffix(path: str) -> None:
    with pytest.raises(ValueError):
        validate_relative_extxyz_path(path)
    with pytest.raises(TypeError):
        validate_relative_extxyz_path(3)  # type: ignore[arg-type]


def test_info_entry_discovers_extxyz_metadata_and_rejects_overrides(extxyz_domain: Path) -> None:
    with pytest.warns(UserWarning, match="Missing optional fields"):
        entry = InfoEntry.from_extxyz(extxyz_domain / "alpha_t_300_run.extxyz", source="unit-test", system_type=["molecule"])

    assert entry.name == "alpha_t_300_run"
    assert entry.domain == "demo_domain"
    assert entry.num_systems == 1
    assert entry.num_frames_per_system == [2]
    assert entry.num_atoms_per_system == [2]
    assert entry.formulas == ["H2"]
    assert entry.has_stress is True
    assert entry.has_other_properties == ["dataset_tag"]

    with pytest.raises(ValueError, match=r"Cannot override automatically detected metadata fields: \['num_systems'\]"):
        InfoEntry.from_extxyz(
            extxyz_domain / "alpha_t_300_run.extxyz",
            source="unit-test",
            system_type=["molecule"],
            num_systems=99,
        )


def test_info_loading_orders_files_and_checks_metadata_count(extxyz_domain: Path, metadata_payload: dict) -> None:
    (extxyz_domain / "metadata.json").write_text(json.dumps(metadata_payload), encoding="utf-8")
    with pytest.warns(UserWarning, match="Missing optional fields"):
        entries = load_info_entries_from_datadir(extxyz_domain)
    assert [entry.filename for entry in entries] == ["alpha_t_300_run.extxyz", "beta_t_600_run.extxyz"]

    (extxyz_domain / "metadata.json").write_text(json.dumps({"info": metadata_payload["info"][:1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Number of data files"):
        load_info_entries_from_datadir(extxyz_domain)


def test_info_entry_requires_reproducibility_fields_and_warns_for_optional() -> None:
    incomplete = required_info()
    incomplete["source"] = ""
    with pytest.raises(ValidationError, match="Missing required fields"):
        InfoEntry(**incomplete)
    with pytest.warns(UserWarning, match="Missing optional fields"):
        assert InfoEntry(**required_info()).name == "sample"


def test_grouped_domain_generates_frame_references_and_validates_group_filenames() -> None:
    first = InfoEntry(**required_info("first.extxyz"), num_frames_per_system=[2, 1])
    second_data = required_info("second.extxyz")
    second_data["name"] = "second"
    second = InfoEntry(**second_data, num_frames_per_system=[1])
    with pytest.warns(UserWarning, match="Missing optional fields"):
        grouped = GroupedDomain(domain="domain", info_entries=[first, second], grouping_strategy="as_specified", groups={"mixed": ["first.extxyz", "second.extxyz"]})
    assert [ref.identity for ref in grouped.load_frame_references_in_groups()["mixed"]] == [
        ("domain", "first.extxyz", 0), ("domain", "first.extxyz", 1), ("domain", "first.extxyz", 2), ("domain", "second.extxyz", 0),
    ]
    with pytest.raises(ValidationError, match="invalid file extension"):
        GroupedDomain(domain="domain", info_entries=[], grouping_strategy="all", groups={"bad": ["not-extxyz.xyz"]})
    with pytest.raises(ValidationError, match="not present in info_entries"):
        GroupedDomain(domain="domain", info_entries=[first], grouping_strategy="all", groups={"missing": ["second.extxyz"]})
    with pytest.raises(ValidationError, match="occurs more than once"):
        GroupedDomain(domain="domain", info_entries=[first], grouping_strategy="all", groups={"one": ["first.extxyz"], "two": ["first.extxyz"]})
    with pytest.raises(ValidationError, match="occurs more than once"):
        GroupedDomain(domain="domain", info_entries=[first], grouping_strategy="all", groups={"one": ["first.extxyz", "first.extxyz"]})


def test_split_schemas_provide_nested_sets_and_reject_invalid_partitions() -> None:
    refs = references()
    trajectory = TrainValSplitTrajectory(method="random", seed=4, requested_train_sizes=[1, 2], selected_frames=refs[:3], additional_trainval_frames=[])
    assert trajectory.get_train_set(1) == refs[:2]
    assert trajectory.get_val_set(1) == refs[2:3]
    valid = SplitGroup(domain="domain", grouping_strategy="all", group_name="all", test_set=[refs[3]], extra_tested_groups=[], test_ratio=0.25, trainval_test_split_seed=1, train_val_split_trajectory=trajectory)
    assert valid.repeat_id == 0

    with pytest.raises(ValidationError, match="duplicate frames"):
        TrainValSplitTrajectory(method="random", seed=4, requested_train_sizes=[1], selected_frames=[refs[0], refs[0]])
    with pytest.raises(ValidationError, match="test_set size"):
        SplitGroup(domain="domain", grouping_strategy="all", group_name="all", test_set=[], extra_tested_groups=[], test_ratio=0.25, trainval_test_split_seed=1, train_val_split_trajectory=trajectory)
    with pytest.raises(ValidationError, match="strictly increasing"):
        EntropyProfile(points=[EntropyProfilePoint(training_size=2, cumulative_entropy=0, information_gain=0), EntropyProfilePoint(training_size=1, cumulative_entropy=0, information_gain=0)])


def test_training_unit_validates_files_immutability_and_movable_root(tmp_path: Path) -> None:
    root = tmp_path / "sets"
    root.mkdir()
    for name in ("train.extxyz", "val.extxyz", "test.extxyz"):
        (root / name).write_text("", encoding="utf-8")
    unit = TrainingUnit(domain="domain", grouping_strategy="all", group_name="all", method="random", repeat_id=0, n_train=1, train_set="train.extxyz", val_set="val.extxyz", test_sets=["test.extxyz"], root_path=root)
    with pytest.raises(ValidationError):
        unit.n_train = 2
    moved_root = tmp_path / "moved"
    moved_root.mkdir()
    for name in ("train.extxyz", "val.extxyz", "test.extxyz"):
        (moved_root / name).write_text("", encoding="utf-8")
    unit.root_path = moved_root
    assert unit.root_path == moved_root
    with pytest.raises(ValidationError, match="does not exist"):
        TrainingUnit(domain="domain", grouping_strategy="all", group_name="all", method="random", repeat_id=0, n_train=1, train_set="missing.extxyz", test_sets=[], root_path=root)
