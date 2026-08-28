"""Focused tests for persisted schemas and metadata discovery."""
from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar, Self
from uuid import UUID

import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from monty.serialization import dumpfn, loadfn
from pydantic import ValidationError, model_validator

from temper.schemas.group import GroupedDomain
from temper.schemas.base import ManagedIdentityModel, MSONableModel
from temper.schemas.info import InfoEntry, load_info_entries_from_datadir
from temper.schemas.split import (
    SplitGroup,
    TrainValSplitTrajectory,
)
from temper.schemas.entropy import EntropyProfilePoint, EntropyProfile
from temper.schemas.frame_refrence import FrameReference
from temper.schemas.train_unit import TrainingUnit
from temper.schemas.utils import (
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


class NestedIdentitySource(MSONableModel):
    selected: int
    ignored: int = 0


class DeclarativeIdentityModel(ManagedIdentityModel):
    _IDENTITY_FIELD_NAME: ClassVar[str] = "record_id"
    _IDENTITY_SOURCE_FIELDS: ClassVar[tuple[str, ...]] = (
        "name",
        "nested.selected",
        "must_be_positive",
    )
    _IDENTITY_NAMESPACE: ClassVar[UUID] = UUID(
        "86ff3ebd-664c-59af-bf36-36a30fe49d63"
    )
    _IDENTITY_SCHEMA: ClassVar[str] = "temper.test-record.v1"
    _IDENTITY_LABEL: ClassVar[str] = "test record"

    name: str
    nested: NestedIdentitySource
    must_be_positive: int
    record_id: UUID | None = None

    def _validate_before_identity(self) -> None:
        if self.must_be_positive <= 0:
            raise ValueError("must_be_positive must be positive")


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
        dumpfn(model, path, indent=2)
        assert loadfn(path) == model
        if isinstance(model, SplitGroup):
            serialized = json.loads(path.read_text(encoding="utf-8"))
            assert serialized["split_id"] == str(model.split_id)


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


def test_split_schemas_provide_nested_sets_and_reject_invalid_partitions(
    tmp_path: Path,
) -> None:
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
    infinite = EntropyProfilePoint(
        training_size=1,
        cumulative_entropy=float("inf"),
        information_gain=float("inf"),
    )
    assert infinite.cumulative_entropy == infinite.information_gain == float("inf")
    infinite_profile_path = tmp_path / "infinite-entropy-profile.json"
    dumpfn(EntropyProfile(points=[infinite]), infinite_profile_path, indent=2)
    restored_infinite_profile = loadfn(infinite_profile_path)
    assert restored_infinite_profile.points[0] == infinite
    with pytest.raises(ValidationError, match="must not be NaN"):
        EntropyProfilePoint(
            training_size=1,
            cumulative_entropy=float("nan"),
            information_gain=0.0,
        )


def test_split_identity_is_deterministic_and_tracks_split_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refs = references()
    trajectory = TrainValSplitTrajectory(
        method="random",
        seed=4,
        requested_train_sizes=[1, 2],
        selected_frames=refs[:3],
        additional_trainval_frames=[],
    )
    split = SplitGroup(
        domain="domain",
        grouping_strategy="all",
        group_name="all",
        test_set=[refs[3]],
        extra_tested_groups=["beta", "alpha"],
        test_ratio=0.25,
        trainval_test_split_seed=1,
        train_val_split_trajectory=trajectory,
    )

    reconstructed = SplitGroup.model_validate(split.model_dump())
    reordered_cross_tests = split.model_copy(
        update={"extra_tested_groups": ["alpha", "beta"]}
    )
    different_seed = split.model_copy(update={"trainval_test_split_seed": 2})

    assert "split_id" in SplitGroup.model_fields
    assert split.split_id.version == 5
    assert split.split_id == UUID("19400556-c9e7-57f4-aed3-b16101ff3840")
    assert reconstructed.split_id == split.split_id
    assert reordered_cross_tests.split_id == split.split_id
    assert different_seed.split_id != split.split_id

    original_id = split.split_id
    split.trainval_test_split_seed = 2
    assert split.split_id != original_id
    changed_id = split.split_id
    with pytest.raises(ValidationError, match="test_ratio"):
        split.test_ratio = 0
    assert split.test_ratio == 0.25
    assert split.split_id == changed_id

    with pytest.raises(AttributeError, match="system-managed"):
        split.split_id = original_id
    with pytest.raises(AttributeError):
        split.test_set.append(refs[0])  # type: ignore[attr-defined]

    before_membership_change = split.split_id
    split.test_set = [
        FrameReference(
            domain="domain",
            filename="sample.extxyz",
            frame_index=4,
        )
    ]
    assert isinstance(split.test_set, tuple)
    assert split.split_id != before_membership_change

    before_entropy_change = split.split_id
    split.train_val_split_trajectory = trajectory.model_copy(update={
        "entropy_profile": EntropyProfile(points=[
            EntropyProfilePoint(
                training_size=1,
                cumulative_entropy=1.0,
                information_gain=1.0,
            )
        ])
    })
    assert split.split_id == before_entropy_change

    serialized = split.model_dump(mode="json")
    serialized["split_id"] = str(original_id)
    with pytest.raises(ValidationError, match="does not match split contents"):
        SplitGroup.model_validate(serialized)

    stored_id = split.split_id

    def unexpected_recomputation(_: SplitGroup) -> UUID:
        raise AssertionError("stored split_id should be reused")

    monkeypatch.setattr(SplitGroup, "_compute_identity", unexpected_recomputation)
    assert split.split_id == stored_id
    assert split.model_dump(mode="json")["split_id"] == str(stored_id)


def test_training_unit_validates_files_identity_updates_and_movable_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "sets"
    domain_root = root / "domain"
    domain_root.mkdir(parents=True)
    for name in ("train.extxyz", "val.extxyz", "test.extxyz"):
        (domain_root / name).write_text("", encoding="utf-8")
    split_id = UUID("70f7b0c5-1894-5964-8e7c-89454fb0f52a")
    unit = TrainingUnit(domain="domain", grouping_strategy="all", group_name="all", method="random", repeat_id=0, n_train=1, split_id=split_id, train_set="train.extxyz", val_set="val.extxyz", test_sets=["test.extxyz"], root_path=root)
    original_training_unit_id = unit.training_unit_id
    assert "training_unit_id" in TrainingUnit.model_fields
    assert unit.training_unit_id.version == 5
    assert unit.training_unit_id == UUID("83586d3e-01ff-5bd6-b440-be69cfb8b803")
    unit.n_train = 2
    assert unit.n_train == 2
    assert unit.training_unit_id != original_training_unit_id
    updated_training_unit_id = unit.training_unit_id
    with pytest.raises(ValidationError):
        unit.n_train = 0
    assert unit.n_train == 2
    assert unit.training_unit_id == updated_training_unit_id
    with pytest.raises(AttributeError, match="system-managed"):
        unit.training_unit_id = original_training_unit_id
    with pytest.raises(AttributeError):
        unit.test_sets.append("other.extxyz")  # type: ignore[attr-defined]
    moved_root = tmp_path / "moved"
    moved_domain_root = moved_root / "domain"
    moved_domain_root.mkdir(parents=True)
    for name in ("train.extxyz", "val.extxyz", "test.extxyz"):
        (moved_domain_root / name).write_text("", encoding="utf-8")
    unit.root_path = moved_root
    assert unit.root_path == moved_root
    assert unit.training_unit_id == updated_training_unit_id

    manifest = tmp_path / "training_unit.json"
    dumpfn(unit, manifest, indent=2)
    serialized = json.loads(manifest.read_text(encoding="utf-8"))
    assert serialized["split_id"] == str(split_id)
    assert serialized["training_unit_id"] == str(updated_training_unit_id)
    loaded = loadfn(manifest)
    assert loaded.split_id == split_id
    assert loaded.training_unit_id == updated_training_unit_id

    other_split = unit.model_copy(
        update={"split_id": UUID("fcb30cc1-5e4c-5bb4-9c08-e932938b3c50")}
    )
    assert other_split.training_unit_id != unit.training_unit_id

    legacy_unit = TrainingUnit(domain="domain", grouping_strategy="all", group_name="all", method="random", repeat_id=0, n_train=1, train_set="train.extxyz", val_set="val.extxyz", test_sets=["test.extxyz"], root_path=moved_root)
    assert legacy_unit.split_id is None
    legacy_payload = legacy_unit.model_dump()
    legacy_payload.pop("training_unit_id")
    assert TrainingUnit.model_validate(legacy_payload).training_unit_id == legacy_unit.training_unit_id

    tampered_payload = unit.model_dump(mode="json")
    tampered_payload["n_train"] = 3
    with pytest.raises(ValidationError, match="does not match training-unit contents"):
        TrainingUnit.model_validate(tampered_payload)
    with pytest.raises(ValidationError, match="does not exist"):
        TrainingUnit(domain="domain", grouping_strategy="all", group_name="all", method="random", repeat_id=0, n_train=1, train_set="missing.extxyz", test_sets=[], root_path=root)

    stored_id = unit.training_unit_id

    def unexpected_recomputation(_: TrainingUnit) -> UUID:
        raise AssertionError("stored training_unit_id should be reused")

    monkeypatch.setattr(
        TrainingUnit,
        "_compute_identity",
        unexpected_recomputation,
    )
    assert unit.training_unit_id == stored_id
    assert unit.model_dump(mode="json")["training_unit_id"] == str(stored_id)


def test_managed_identity_subclass_is_declarative_and_uses_validation_hook() -> None:
    record = DeclarativeIdentityModel(
        name="alpha",
        nested=NestedIdentitySource(selected=1, ignored=10),
        must_be_positive=1,
    )

    assert "_identity_payload" not in DeclarativeIdentityModel.__dict__
    assert "_finalize_managed_identity" not in DeclarativeIdentityModel.__dict__
    assert record._identity_payload() == {
        "identity_schema": "temper.test-record.v1",
        "name": "alpha",
        "nested": {"selected": 1},
        "must_be_positive": 1,
    }
    assert record.model_dump()["record_id"] == str(record.record_id)

    original_id = record.record_id
    record.nested = NestedIdentitySource(selected=1, ignored=20)
    assert record.record_id == original_id
    record.nested = NestedIdentitySource(selected=2, ignored=20)
    assert record.record_id != original_id

    valid_id = record.record_id
    with pytest.raises(ValidationError, match="must_be_positive"):
        record.must_be_positive = 0
    assert record.must_be_positive == 1
    assert record.record_id == valid_id

    with pytest.raises(ValidationError, match="must_be_positive"):
        DeclarativeIdentityModel(
            name="invalid",
            nested=NestedIdentitySource(selected=1),
            must_be_positive=0,
        )


def test_managed_identity_rejects_invalid_developer_configuration() -> None:
    with pytest.raises(TypeError, match="missing model field 'missing'"):

        class MissingIdentitySource(ManagedIdentityModel):
            _IDENTITY_FIELD_NAME: ClassVar[str] = "record_id"
            _IDENTITY_SOURCE_FIELDS: ClassVar[tuple[str, ...]] = ("missing",)
            _IDENTITY_NAMESPACE: ClassVar[UUID] = UUID(
                "cc8f743e-8371-5780-8b24-c11fb4d93510"
            )
            _IDENTITY_SCHEMA: ClassVar[str] = "temper.invalid-record.v1"
            _IDENTITY_LABEL: ClassVar[str] = "invalid record"

            value: int
            record_id: UUID | None = None

    with pytest.raises(TypeError, match="_validate_before_identity"):

        class InvalidAfterValidator(ManagedIdentityModel):
            _IDENTITY_FIELD_NAME: ClassVar[str] = "record_id"
            _IDENTITY_SOURCE_FIELDS: ClassVar[tuple[str, ...]] = ("value",)
            _IDENTITY_NAMESPACE: ClassVar[UUID] = UUID(
                "bc7f0ca1-c4b5-52b1-863e-b3179b20984f"
            )
            _IDENTITY_SCHEMA: ClassVar[str] = "temper.invalid-record.v1"
            _IDENTITY_LABEL: ClassVar[str] = "invalid record"

            value: int
            record_id: UUID | None = None

            @model_validator(mode="after")
            def validate_value(self) -> Self:
                return self
