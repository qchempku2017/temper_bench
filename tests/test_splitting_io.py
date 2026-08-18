"""Tests for safe reference reconstruction and deterministic extxyz exports."""
from __future__ import annotations

from pathlib import Path

import pytest
from ase.io import read, write

from conftest import make_frame
from src.temper.schemas.split import SplitGroup, TrainValSplitTrajectory
from src.temper.schemas.entropy import EntropyProfilePoint, EntropyProfile
from src.temper.schemas.frame_refrence import FrameReference
from src.temper.splitting.io import FrameReferenceResolver, build_export_filename, load_frames_from_references, write_all_sets_in_split_group_to_extxyz


def _refs() -> list[FrameReference]:
    return [FrameReference(domain="demo", filename="frames.extxyz", frame_index=index) for index in range(10)]


def _split_group() -> SplitGroup:
    refs = _refs()
    return SplitGroup(
        domain="demo", grouping_strategy="all", group_name="main", test_set=refs[8:],
        extra_tested_groups=[], test_ratio=0.2, trainval_test_split_seed=1, repeat_id=0,
        train_val_split_trajectory=TrainValSplitTrajectory(
            method="random", seed=2, requested_train_sizes=[2, 4], selected_frames=refs[:4],
            additional_trainval_frames=refs[4:8], entropy_profile=EntropyProfile(points=[
                EntropyProfilePoint(training_size=1, cumulative_entropy=1.0, information_gain=1.0),
                EntropyProfilePoint(training_size=4, cumulative_entropy=2.0, information_gain=1.0),
            ]),
        ),
    )


def _other_split_group() -> SplitGroup:
    refs = [FrameReference(domain="demo", filename="other.extxyz", frame_index=index) for index in range(10)]
    return SplitGroup(
        domain="demo", grouping_strategy="all", group_name="other", test_set=refs[8:], extra_tested_groups=[],
        test_ratio=0.2, trainval_test_split_seed=1, repeat_id=0,
        train_val_split_trajectory=TrainValSplitTrajectory(
            method="random", seed=2, requested_train_sizes=[2, 4], selected_frames=refs[:4],
            additional_trainval_frames=refs[4:8],
        ),
    )


def test_resolver_cache_preserves_reference_order_and_rejects_escapes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "demo"
    source.mkdir()
    filename = source / "frames.extxyz"
    write(filename, [make_frame("H", -1.0, "first"), make_frame("He", -2.0, "second")], format="extxyz")
    import src.temper.splitting.io as io_module
    original_read = io_module.read
    calls: list[Path] = []

    def counted_read(path: Path, index: str):
        calls.append(Path(path))
        return original_read(path, index=index)

    monkeypatch.setattr(io_module, "read", counted_read)
    refs = [FrameReference(domain="demo", filename="frames.extxyz", frame_index=1), FrameReference(domain="demo", filename="frames.extxyz", frame_index=0)]
    frames, resolver = load_frames_from_references(refs, tmp_path)
    assert isinstance(resolver, FrameReferenceResolver)
    again, same_resolver = load_frames_from_references(refs, tmp_path, resolver)
    assert [frame.info["dataset_tag"] for frame in frames] == ["second", "first"]
    assert [frame.info["dataset_tag"] for frame in again] == ["second", "first"]
    assert same_resolver is resolver
    assert calls == [filename.resolve()]  # Same file should have been cached and not read twice.
    unsafe = FrameReference.model_construct(domain="../outside", filename="frames.extxyz", frame_index=0)
    with pytest.raises(ValueError, match="single safe"):
        resolver.resolve_source_path(unsafe)


def test_export_writes_exact_named_train_validation_and_test_sets(tmp_path: Path) -> None:
    source = tmp_path / "source" / "demo"
    source.mkdir(parents=True)
    write(source / "frames.extxyz", [make_frame("H", -float(index), str(index)) for index in range(10)], format="extxyz")
    output = tmp_path / "out"
    units, resolver = write_all_sets_in_split_group_to_extxyz(
        _split_group(), root_path=source.parent, output_path=output,
        write_validation=True, write_extra_tests=False,
    )  # TODO: add test cases where extra_tests are correctly treated.
    assert len(units) == 2
    assert [(unit.n_train, unit.val_set is not None) for unit in units] == [(2, True), (4, True)]
    assert [len(read(output / unit.train_set, index=":")) for unit in units] == [2, 4]
    assert [len(read(output / unit.val_set, index=":")) for unit in units if unit.val_set] == [6, 4]
    assert len(read(output / units[0].test_sets[0], index=":")) == 2
    assert all("__n" in filename and filename.endswith(".extxyz") for unit in units for filename in [unit.train_set, *unit.test_sets])
    assert resolver.root_path == source.parent.resolve()
    assert build_export_filename("d /", "g", None, "random", "train", 2, 3) == "d____unknown_grouping__g__random__train__n2__repeat3.extxyz"
    with pytest.raises(ValueError, match="role"):
        build_export_filename("d", "g", "s", "m", "other", 1, 0)


def test_export_includes_cross_tests_only_when_requested(tmp_path: Path) -> None:
    source = tmp_path / "source" / "demo"
    source.mkdir(parents=True)
    for filename in ("frames.extxyz", "other.extxyz"):
        write(source / filename, [make_frame("H", -float(index), f"{filename}-{index}") for index in range(10)], format="extxyz")
    main = _split_group().model_copy(update={"extra_tested_groups": ["other"]})
    other = _other_split_group()

    without_extra, _ = write_all_sets_in_split_group_to_extxyz(
        main, root_path=source.parent, output_path=tmp_path / "without-extra", write_extra_tests=False,
    )
    assert all(len(unit.test_sets) == 1 for unit in without_extra)

    with_extra, _ = write_all_sets_in_split_group_to_extxyz(
        main, root_path=source.parent, output_path=tmp_path / "with-extra", write_extra_tests=True,
        all_split_groups=[main, other],
    )
    assert all(len(unit.test_sets) == 2 for unit in with_extra)
    assert any("__other__" in filename for filename in with_extra[0].test_sets)
