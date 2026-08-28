"""Deterministic unit tests for split partitioning, selectors, and orchestration."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from ase.io import write
from monty.serialization import dumpfn, loadfn

from temper.schemas.group import GroupedDomain
from temper.schemas.info import InfoEntry
from temper.schemas.frame_refrence import FrameReference
from temper.splitting import selectors
from temper.splitting.quests_adapter import QuestsAdapterConfig, QuestsDescriptorsStorage
from temper.splitting.split import partition_trainval_test, split_grouped_domain
from temper.schemas.split import SplitConfig
from temper.splitting.utils import get_requested_train_sizes_from_ratios
from conftest import make_frame


def _references(count: int) -> list[FrameReference]:
    return [FrameReference(domain="demo", filename="frames.extxyz", frame_index=index) for index in range(count)]


def _storage(count: int) -> QuestsDescriptorsStorage:
    return QuestsDescriptorsStorage(
        values=np.arange(count * 2, dtype=float).reshape(count, 2),
        frame_offsets=tuple(range(count + 1)),
        quests_adapter_config=QuestsAdapterConfig(device="cpu"),
    )  # `count` structures, each has one atom.


class _EntropyAdapter:
    def __init__(self, config: QuestsAdapterConfig) -> None:
        self.config = config

    def get_entropy(self, descriptors: np.ndarray) -> float:
        return float(descriptors.sum())


def test_requested_sizes_and_partition_are_exact_and_deterministic() -> None:
    assert get_requested_train_sizes_from_ratios(10, [0.8, 0.2], max_train_size=4) == [1, 4]
    with pytest.raises(ValueError, match="too small"):
        get_requested_train_sizes_from_ratios(3, [0.1])

    pool = _references(10)
    trainval, test, train_positions, test_positions = partition_trainval_test(pool, seed=7, test_ratio=0.2)
    assert test_positions == [6, 8]
    assert train_positions == [0, 1, 2, 3, 4, 5, 7, 9]
    assert [frame.frame_index for frame in trainval] == train_positions
    assert [frame.frame_index for frame in test] == test_positions
    assert set(train_positions).isdisjoint(test_positions)
    with pytest.raises(ValueError, match="duplicate"):
        partition_trainval_test([pool[0], pool[0]], seed=1, test_ratio=0.5)


def test_random_selector_maps_full_pool_indices_and_nested_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(selectors, "QuestsAdapter", _EntropyAdapter)
    selector = selectors.RandomIndicesSelector(
        _storage(8), [0, 2, 3, 5, 7], requested_train_ratios=[0.4, 0.8],
        max_train_size=4, seed=14, num_selected_per_step=1,
    )
    selected, remaining, profile = selector.run()
    assert selector.requested_train_sizes == [2, 4]
    assert len(selected) == 4
    assert len(set(selected)) == 4
    assert set(selected).issubset({0, 2, 3, 5, 7})
    assert sorted(selected + remaining) == [0, 2, 3, 5, 7]
    assert [point.training_size for point in profile.points] == [1, 2, 3, 4]


def test_selector_preserves_infinite_entropy_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    class InfiniteEntropyAdapter(_EntropyAdapter):
        def get_entropy(self, descriptors: np.ndarray) -> float:
            return float("inf")

    monkeypatch.setattr(selectors, "QuestsAdapter", InfiniteEntropyAdapter)
    selector = selectors.RandomIndicesSelector(
        _storage(5), [0, 1, 2, 3, 4], requested_train_ratios=[0.6],
        max_train_size=3, seed=14, num_selected_per_step=1,
    )
    _, _, profile = selector.run()
    assert [point.cumulative_entropy for point in profile.points] == [
        float("inf"),
        float("inf"),
        float("inf"),
    ]
    assert [point.information_gain for point in profile.points] == [
        float("inf"),
        float("inf"),
        float("inf"),
    ]


def test_selector_helpers_validate_pool_membership_and_rank_entropy_gain() -> None:
    storage = _storage(4)

    class Adapter:
        def delta_entropy(self, candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
            return candidate[:, 0]

    assert selectors.greedy_select_frame_indices_by_entropy_gain(storage, Adapter(), [0], 2, [0, 1, 2, 3]) == [3, 2]

    class InfiniteAdapter:
        def delta_entropy(self, candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
            deltas = candidate[:, 0].copy()
            deltas[0] = float("inf")
            return deltas

    assert selectors.greedy_select_frame_indices_by_entropy_gain(
        storage, InfiniteAdapter(), [0], 2, [0, 1, 2, 3]
    ) == [1, 3]
    with pytest.raises(ValueError, match="not all within"):
        selectors.select_frame_indices_at_random([4], 1, [0, 1], np.random.default_rng(1))
    with pytest.raises(ValueError, match="Unknown"):
        selectors.selector_class_factory("other")


def test_split_config_and_orchestration_preserve_partitions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    domain_dir = tmp_path / "demo"
    domain_dir.mkdir()
    write(domain_dir / "frames.extxyz", [make_frame("H", -float(index), str(index)) for index in range(10)], format="extxyz")
    with pytest.warns(UserWarning, match="Missing optional fields"):
        info = InfoEntry(
            name="demo", source="test", domain="demo", filename="frames.extxyz",
            system_type=["atom"], num_frames_per_system=[10],
        )
    grouped = GroupedDomain(domain="demo", info_entries=[info], grouping_strategy="all", groups={"all": ["frames.extxyz"]})

    class ComputeAdapter(_EntropyAdapter):
        def resolve_device(self) -> str:
            return "gpu"

        def compute_descriptors(self, frames: list) -> QuestsDescriptorsStorage:
            return _storage(len(frames))

    import temper.splitting.split as split_module
    monkeypatch.setattr(split_module, "QuestsAdapter", ComputeAdapter)
    monkeypatch.setattr(selectors, "QuestsAdapter", _EntropyAdapter)
    config = SplitConfig(root_path=tmp_path, split_repeats=1, seed=7, test_ratio=0.2, requested_train_ratios=[0.25, 0.5], max_train_size=4, train_val_split_method="random")
    with caplog.at_level(logging.INFO, logger="temper.splitting.split"):
        result = split_grouped_domain(grouped, config)
    assert len(result) == 1
    assert any(
        "entropy=GPU (auto-selected because torch.cuda.is_available() returned "
        "True; PyTorch kernel compatibility with the GPU architecture was not "
        "checked)" in record.getMessage()
        for record in caplog.records
    )
    split = result[0]
    trajectory = split.train_val_split_trajectory
    assert len(split.test_set) == 2
    assert trajectory.requested_train_sizes == (2, 4)
    assert len(trajectory.selected_frames) == 4
    assert len(trajectory.additional_trainval_frames) == 4
    assert {ref.identity for ref in split.test_set}.isdisjoint(ref.identity for ref in trajectory.selected_frames + trajectory.additional_trainval_frames)
    assert split.train_val_split_trajectory.entropy_profile is not None


def test_cross_test_configuration_supports_automatic_tests_and_specified_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    domain_dir = tmp_path / "demo"
    domain_dir.mkdir()
    for filename in ("left.extxyz", "right.extxyz"):
        write(
            domain_dir / filename,
            [make_frame("H", -float(index), f"{filename}-{index}") for index in range(10)],
            format="extxyz",
        )
    infos = [
        InfoEntry(
            name=filename.removesuffix(".extxyz"), source="test", domain="demo",
            filename=filename, system_type=["atom"], num_frames_per_system=[10],
        )
        for filename in ("left.extxyz", "right.extxyz")
    ]

    class ComputeAdapter(_EntropyAdapter):
        def compute_descriptors(self, frames: list) -> QuestsDescriptorsStorage:
            return _storage(len(frames))

    import temper.splitting.split as split_module
    monkeypatch.setattr(split_module, "QuestsAdapter", ComputeAdapter)
    monkeypatch.setattr(selectors, "QuestsAdapter", _EntropyAdapter)
    config = SplitConfig(
        root_path=tmp_path, split_repeats=1, seed=7,
        test_ratio=0.2, requested_train_ratios=[0.5],
        max_train_size=4, train_val_split_method="random",
    )
    groups = {"left": ["left.extxyz"], "right": ["right.extxyz"]}
    automatic = GroupedDomain(
        domain="demo", info_entries=infos, grouping_strategy="as_specified", groups=groups,
        add_extra_cross_tests=True,
    )
    automatic_splits = split_grouped_domain(automatic, config)
    assert {split.group_name: split.extra_tested_groups for split in automatic_splits} == {
        "left": ("right",), "right": ("left",),
    }

    specified = GroupedDomain(
        domain="demo", info_entries=infos, grouping_strategy="as_specified", groups=groups,
        add_extra_cross_tests=True, specify_cross_tests={"left": ["left", "right", "right"]},
    )
    specified_splits = split_grouped_domain(specified, config)
    assert {split.group_name: split.extra_tested_groups for split in specified_splits} == {
        "left": ("right",), "right": (),
    }  # Automatically deduplicated.


def test_split_config_derives_and_round_trips_complete_reproducible_seed_lists(
    tmp_path: Path,
) -> None:
    first = SplitConfig(split_repeats=2, seed=42)
    second = SplitConfig(split_repeats=2, seed=42)
    assert len(first.trainval_test_split_seeds) == len(first.train_val_split_seeds) == 2
    assert first.trainval_test_split_seeds == second.trainval_test_split_seeds
    assert first.train_val_split_seeds == second.train_val_split_seeds
    exact = SplitConfig(
        root_path=tmp_path / "data",
        output_path=tmp_path / "results",
        split_repeats=2,
        seed=99,
        trainval_test_split_seeds=[123, 456],
        train_val_split_seeds=[789, 101112],
    )
    path = tmp_path / "split_config_reproduce.json"
    dumpfn(exact, path, indent=2)
    restored = loadfn(path)
    assert restored == exact
    assert restored.trainval_test_split_seeds == [123, 456]
    assert restored.train_val_split_seeds == [789, 101112]
    with pytest.raises(ValueError, match="Length"):
        SplitConfig(seed=42, split_repeats=2, trainval_test_split_seeds=[1], train_val_split_seeds=[2, 3])
    with pytest.raises(ValueError, match="non-negative integer"):
        SplitConfig(split_repeats=1, seed=-1)


def test_split_config_resolves_sorted_direct_domain_children_immutably(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    for domain in ("zeta", "alpha"):
        domain_path = data_root / domain
        domain_path.mkdir()
        (domain_path / "metadata.json").write_text("{}", encoding="utf-8")

    (data_root / "without-metadata").mkdir()
    nested_domain = data_root / "container" / "nested"
    nested_domain.mkdir(parents=True)
    (nested_domain / "metadata.json").write_text("{}", encoding="utf-8")
    (data_root / "metadata.json").write_text("{}", encoding="utf-8")

    config = SplitConfig(
        root_path=data_root,
        domains=None,
        split_repeats=1,
        seed=7,
    )
    resolved = config.resolve_domains()

    assert config.domains is None
    assert resolved is not config
    assert resolved.domains == ["alpha", "zeta"]
    assert resolved.trainval_test_split_seeds == config.trainval_test_split_seeds
    assert resolved.train_val_split_seeds == config.train_val_split_seeds
    assert resolved.resolve_domains() is resolved

    added_domain = data_root / "beta"
    added_domain.mkdir()
    (added_domain / "metadata.json").write_text("{}", encoding="utf-8")

    assert config.resolve_domains().domains == ["alpha", "beta", "zeta"]
    assert resolved.domains == ["alpha", "zeta"]


def test_split_config_resolve_domains_skips_io_for_explicit_domains(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing"
    explicit = SplitConfig(
        root_path=missing_root,
        domains=["selected"],
        split_repeats=1,
        seed=7,
    )
    empty = SplitConfig(
        root_path=missing_root,
        domains=[],
        split_repeats=1,
        seed=7,
    )

    assert explicit.resolve_domains() is explicit
    assert empty.resolve_domains() is empty

    automatic = SplitConfig(
        root_path=missing_root,
        domains=None,
        split_repeats=1,
        seed=7,
    )
    with pytest.raises(ValueError, match="Data root is not a directory"):
        automatic.resolve_domains()
