"""Tests for QUESTS boundary behavior, defaults, package exports, and CLI wiring."""
from __future__ import annotations

import importlib
import inspect
import sys
import warnings
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from temper.splitting.quests_adapter import QuestsAdapter, QuestsAdapterConfig, QuestsDescriptorsStorage, QuestsNumericalError, compute_information_gain_per_candidate_frame


def test_installed_quests_cpu_api_matches_adapter_contract() -> None:
    """Pin every QUESTS CPU symbol and keyword consumed by the adapter."""
    import quests.descriptor as descriptor_module
    import quests.entropy as entropy_module

    assert tuple(inspect.signature(descriptor_module.get_descriptors).parameters) == (
        "dset",
        "k",
        "cutoff",
        "concat",
        "dtype",
    )
    assert tuple(inspect.signature(entropy_module.entropy).parameters) == (
        "x",
        "h",
        "batch_size",
    )
    assert tuple(inspect.signature(entropy_module.delta_entropy).parameters) == (
        "y",
        "x",
        "h",
        "batch_size",
    )


def test_installed_quests_gpu_api_matches_adapter_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate the optional GPU module's API without requiring torch."""
    import quests

    fake_torch = ModuleType("torch")
    setattr(fake_torch, "Tensor", object)
    setattr(fake_torch, "tensor", object)

    with monkeypatch.context() as module_patch:
        module_patch.delattr(quests, "gpu", raising=False)
        module_patch.delitem(sys.modules, "quests.gpu", raising=False)
        module_patch.delitem(sys.modules, "quests.gpu.entropy", raising=False)
        module_patch.delitem(sys.modules, "quests.gpu.matrix", raising=False)
        module_patch.setitem(sys.modules, "torch", fake_torch)
        entropy_module = importlib.import_module("quests.gpu.entropy")

        assert tuple(inspect.signature(entropy_module.entropy).parameters) == (
            "x",
            "h",
            "batch_size",
            "device",
        )
        assert tuple(inspect.signature(entropy_module.delta_entropy).parameters) == (
            "x",
            "y",
            "h",
            "batch_size",
            "device",
        )


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
        def get_descriptors(frames, *, k, cutoff, concat, dtype):
            assert (k, cutoff, concat, dtype) == (32, 5.0, True, "float64")
            calls.append(len(frames))
            return np.full((sum(len(frame) for frame in frames), 2), len(calls), dtype=np.float64)

    class Entropy:
        @staticmethod
        def entropy(matrix, *, h, batch_size):
            assert (h, batch_size) == (0.015, 4000)
            return 1.25

        @staticmethod
        def delta_entropy(candidate, reference, *, h, batch_size):
            assert (h, batch_size) == (0.015, 4000)
            return np.arange(len(candidate), dtype=float)

    monkeypatch.setattr(adapter, "_configure_numba_cpu_threads", lambda: None)
    monkeypatch.setattr(adapter, "_import_cpu_backend", lambda: (Descriptor, Entropy))
    storage = adapter.compute_descriptors([Atoms("H"), Atoms("He"), Atoms("H2")])
    assert calls == [2, 1]  # This is because compute_descriptor_chunk_size has been set to 2.
    assert storage.frame_offsets == (0, 1, 2, 4)
    assert adapter.get_entropy(np.ones((2, 2))) == 1.25
    assert adapter.delta_entropy(np.ones((2, 2)), np.ones((1, 2))).tolist() == [0.0, 1.0]
    monkeypatch.setattr(adapter, "_import_cpu_backend", lambda: (Descriptor, SimpleNamespace(entropy=lambda *_args, **_kwargs: float("nan"))))
    with pytest.raises(QuestsNumericalError, match="non-finite"):
        adapter.get_entropy(np.ones((1, 2)))


def test_adapter_uses_quests_gpu_entropy_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise both GPU calls without requiring torch or an accelerator."""
    adapter = QuestsAdapter(QuestsAdapterConfig(device="gpu", gpu_device="cuda:2"))

    class TensorResult:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def item(self):
            return self.value

        def numpy(self):
            return self.value

    class GpuEntropy:
        @staticmethod
        def entropy(matrix, *, h, batch_size, device):
            assert matrix == (2, 3)
            assert (h, batch_size, device) == (0.015, 4000, "cuda:2")
            return TensorResult(1.5)

        @staticmethod
        def delta_entropy(candidate, reference, *, h, batch_size, device):
            assert candidate == (2, 3)
            assert reference == (1, 3)
            assert (h, batch_size, device) == (0.015, 4000, "cuda:2")
            return TensorResult(np.array([0.25, 0.75]))

    monkeypatch.setattr(adapter, "resolve_device", lambda: "gpu")
    monkeypatch.setattr(adapter, "_import_gpu_entropy", lambda: GpuEntropy)
    monkeypatch.setattr(adapter, "_to_tensor", lambda matrix: matrix.shape)

    assert adapter.get_entropy(np.ones((2, 3))) == 1.5
    assert adapter.delta_entropy(
        np.ones((2, 3)),
        np.ones((1, 3)),
    ).tolist() == [0.25, 0.75]


def test_auto_device_uses_gpu_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = QuestsAdapter(QuestsAdapterConfig(device="auto"))
    monkeypatch.setattr(adapter, "_assert_gpu_available", lambda: None)
    assert adapter.resolve_device() == "gpu"


def test_auto_device_warns_once_when_falling_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import temper.splitting.quests_adapter as adapter_module
    from temper.splitting.quests_adapter import QuestsUnavailableError
    from temper.logging import BackendFallbackWarning

    adapter_module._WARNED_BACKEND_FALLBACKS.clear()

    def unavailable() -> None:
        raise QuestsUnavailableError("no compatible accelerator")

    first = QuestsAdapter(QuestsAdapterConfig(device="auto"))
    second = QuestsAdapter(QuestsAdapterConfig(device="auto"))
    monkeypatch.setattr(first, "_assert_gpu_available", unavailable)
    monkeypatch.setattr(second, "_assert_gpu_available", unavailable)

    with pytest.warns(BackendFallbackWarning, match="substantially slower"):
        assert first.resolve_device() == "cpu"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert second.resolve_device() == "cpu"
    assert caught == []


def test_defaults_environment_parsing_and_public_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    import temper.utils.defaults as defaults
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
    import temper.grouping as grouping
    assert {"partition_domain_into_groups"}.issubset(grouping.__all__)
    assert defaults.DEFAULT_SPLIT_RESULTS_DIR == "./split_results"
    assert defaults.DEFAULT_GROUPED_DOMAIN_FILE == "grouped_domains.json"
    assert defaults.DEFAULT_SPLIT_GROUPS_FILE == "split_groups.json"
    assert defaults.DEFAULT_TRAINING_UNITS_FILE == "training_units.json"
    importlib.reload(defaults)
