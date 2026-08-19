"""Tests for QUESTS boundary behavior, defaults, package exports, and CLI wiring."""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from src.temper.splitting.quests_adapter import QuestsAdapter, QuestsAdapterConfig, QuestsDescriptorsStorage, QuestsNumericalError, compute_information_gain_per_candidate_frame


def test_descriptor_storage_slices_and_information_gain_aggregate_per_frame() -> None:
    config = QuestsAdapterConfig(device="cpu")
    storage = QuestsDescriptorsStorage(
        values=np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
        frame_offsets=(0, 2, 3, 5), quests_adapter_config=config,
    )

    class Adapter:
        def delta_entropy(self, candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
            assert reference.tolist() == [[1.0], [2.0]]
            return candidate[:, 0]

    assert storage.n_frames == 3
    assert storage.frame_atom_counts == (2, 1, 2)
    assert storage.get_multiple_frames([2, 0]).ravel().tolist() == [4.0, 5.0, 1.0, 2.0]
    assert compute_information_gain_per_candidate_frame(storage, Adapter(), [0], [1, 2]).tolist() == [3.0, 9.0]
    with pytest.raises(ValueError, match="last frame offset"):
        QuestsDescriptorsStorage(np.ones((2, 1)), (0, 1, 3), config)


def test_adapter_chunks_descriptors_and_validates_entropy_results(monkeypatch: pytest.MonkeyPatch) -> None:
    config = QuestsAdapterConfig(device="cpu", compute_descriptor_chunk_size=2)
    adapter = QuestsAdapter(config)
    calls: list[int] = []

    class Descriptor:
        @staticmethod
        def get_descriptors(frames, **kwargs):
            calls.append(len(frames))
            return np.full((sum(len(frame) for frame in frames), 2), len(calls), dtype=np.float32)

    class Entropy:
        @staticmethod
        def get_entropy(matrix, **kwargs):
            return 1.25

        @staticmethod
        def delta_entropy(candidate, reference, **kwargs):
            return np.arange(len(candidate), dtype=float)

    monkeypatch.setattr(adapter, "_configure_numba_cpu_threads", lambda: None)
    monkeypatch.setattr(adapter, "_import_cpu_backend", lambda: (Descriptor, Entropy))
    storage = adapter.compute_descriptors([Atoms("H"), Atoms("He"), Atoms("H2")])
    assert calls == [2, 1]  # This is because compute_descriptor_chunk_size has been set to 2.
    assert storage.frame_offsets == (0, 1, 2, 4)
    assert adapter.get_entropy(np.ones((2, 2))) == 1.25
    assert adapter.delta_entropy(np.ones((2, 2)), np.ones((1, 2))).tolist() == [0.0, 1.0]
    monkeypatch.setattr(adapter, "_import_cpu_backend", lambda: (Descriptor, SimpleNamespace(get_entropy=lambda *_args, **_kwargs: float("nan"))))
    with pytest.raises(QuestsNumericalError, match="non-finite"):
        adapter.get_entropy(np.ones((1, 2)))


def test_auto_device_uses_gpu_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = QuestsAdapter(QuestsAdapterConfig(device="auto"))
    monkeypatch.setattr(adapter, "_assert_gpu_available", lambda: None)
    assert adapter.resolve_device() == "gpu"


def test_defaults_environment_parsing_and_public_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.temper.utils.defaults as defaults
    monkeypatch.setenv("DEFAULT_TEST_RATIO", "0.35")
    monkeypatch.setenv("DEFAULT_MAX_N_TRAIN", "12")
    monkeypatch.setenv("DEFAULT_TRAIN_RATIOS", "0.2, 0.5")
    monkeypatch.setenv("DEFAULT_SPLIT_CONFIG_FILE", "custom-split.yaml")
    defaults = importlib.reload(defaults)
    assert (defaults.DEFAULT_TEST_RATIO, defaults.DEFAULT_MAX_N_TRAIN, defaults.DEFAULT_TRAIN_RATIOS) == (0.35, 12, [0.2, 0.5])
    assert defaults.DEFAULT_SPLIT_CONFIG_FILE == "custom-split.yaml"
    assert defaults._env_float("MISSING_FLOAT", 2.5) == 2.5
    monkeypatch.setenv("INVALID_INT", "two")
    with pytest.raises(ValueError, match="INVALID_INT"):
        defaults._env_int("INVALID_INT", 1)
    import src.temper.grouping as grouping
    assert {"partition_domain_into_groups"}.issubset(grouping.__all__)
    assert defaults.DEFAULT_SPLIT_RESULTS_DIR == "./split_results"
    assert defaults.DEFAULT_GROUPED_DOMAIN_FILE == "grouped_domains.json"
    assert defaults.DEFAULT_SPLIT_GROUPS_FILE == "split_groups.json"
    assert defaults.DEFAULT_TRAINING_UNITS_FILE == "training_units.json"
    importlib.reload(defaults)
