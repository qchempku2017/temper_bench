"""Dataset hashing during construction and metadata-only validation."""

from pathlib import Path

import pytest

from temper.schemas.train_unit import TrainingUnit
from temper.utils.file_hashing import cache_dataset_hashes, file_sha256


@pytest.mark.parametrize("supplied", ["omitted", "empty", "complete"])
def test_construction_finalizes_hashes_before_identity(
    finetune_training_unit: TrainingUnit, supplied: str,
) -> None:
    """Create the same hashed identity with implicit or supplied checksums.

    Parameters
    ----------
    finetune_training_unit : TrainingUnit
        Unit referencing readable training, validation and test datasets.
    supplied : str
        Whether the input omits hashes, supplies an empty mapping, or supplies
        every checksum explicitly.
    """
    source = finetune_training_unit
    names = (*source.test_sets, source.train_set, source.val_set)
    expected = {
        name: file_sha256(source.resolve_local_dataset_path(name))
        for name in names if name is not None
    }
    assert source.dataset_sha256 == expected
    data = source.model_dump(exclude={"training_unit_id", "dataset_sha256"})
    if supplied != "omitted":
        data["dataset_sha256"] = {} if supplied == "empty" else expected.copy()
    unit = TrainingUnit(**data)
    assert unit.dataset_sha256 == expected
    assert unit.training_unit_id == source.training_unit_id
    assert data.get("dataset_sha256", {}) == (
        expected if supplied == "complete" else {}
    )
    assert TrainingUnit.model_validate_json(unit.model_dump_json()) == unit


@pytest.mark.parametrize("with_hashes", [False, True])
def test_metadata_loading_never_reads_original_files(
    finetune_training_unit: TrainingUnit,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_hashes: bool,
) -> None:
    """Preserve supplied or absent hashes when source verification is disabled.

    Parameters
    ----------
    finetune_training_unit : TrainingUnit
        Record supplying valid dataset metadata and checksums.
    tmp_path : Path
        Parent of a deliberately nonexistent dataset root.
    monkeypatch : pytest.MonkeyPatch
        Replaces hashing with a failure if validation tries to read a file.
    with_hashes : bool
        Whether the input record contains checksums and its saved identity.
    """
    def unexpected_hash(path: str | Path) -> str:
        raise AssertionError(f"Metadata loading must not hash {path}")

    monkeypatch.setattr("temper.schemas.train_unit.file_sha256", unexpected_hash)
    data = finetune_training_unit.model_dump()
    data["root_path"] = tmp_path / "unavailable"
    if not with_hashes:
        data.pop("dataset_sha256")
        data.pop("training_unit_id")
    unit = TrainingUnit.model_validate(data, context={"verify_files": False})
    assert unit.dataset_sha256 == (
        finetune_training_unit.dataset_sha256 if with_hashes else {}
    )
    if with_hashes:
        assert unit.training_unit_id == finetune_training_unit.training_unit_id
    assert TrainingUnit.model_validate_json(
        unit.model_dump_json(), context={"verify_files": False}
    ) == unit


@pytest.mark.parametrize("invalid", ["changed", "partial", "malformed"])
def test_supplied_hashes_are_rejected_instead_of_replaced(
    finetune_training_unit: TrainingUnit, invalid: str,
) -> None:
    """Reject incorrect supplied checksums without overwriting caller data.

    Parameters
    ----------
    finetune_training_unit : TrainingUnit
        Record with valid hashes for multiple dataset files.
    invalid : str
        Wrong digest, incomplete filename coverage, or malformed digest text.
    """
    data = finetune_training_unit.model_dump(exclude={"training_unit_id"})
    filename = finetune_training_unit.test_sets[0]
    if invalid == "partial":
        data["dataset_sha256"].pop(filename)
        message = "cover exactly"
    else:
        data["dataset_sha256"][filename] = (
            "0" * 64 if invalid == "changed" else "invalid"
        )
        message = "bytes changed" if invalid == "changed" else "SHA-256"
    before = data["dataset_sha256"].copy()
    with pytest.raises(ValueError, match=message):
        TrainingUnit(**data)
    assert data["dataset_sha256"] == before


def test_construction_reuses_hashes_only_within_one_operation(
    finetune_training_unit: TrainingUnit, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hash shared files once per construction operation and reread afterward.

    Parameters
    ----------
    finetune_training_unit : TrainingUnit
        Record whose datasets are shared by several constructed units.
    monkeypatch : pytest.MonkeyPatch
        Counts file reads during automatic hash calculation.
    """
    calls: list[Path] = []

    def counted_hash(path: str | Path) -> str:
        calls.append(Path(path))
        return file_sha256(path)

    monkeypatch.setattr("temper.schemas.train_unit.file_sha256", counted_hash)
    data = finetune_training_unit.model_dump(
        exclude={"training_unit_id", "dataset_sha256"}
    )
    with cache_dataset_hashes():
        first = TrainingUnit(**data)
        second = TrainingUnit(**(data | {"repeat_id": first.repeat_id + 1}))
    assert first.training_unit_id != second.training_unit_id
    assert len(calls) == len(set(calls)) == len(first.dataset_sha256)
    TrainingUnit(**data)
    assert len(calls) == 2 * len(first.dataset_sha256)
