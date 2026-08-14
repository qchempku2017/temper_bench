"""Focused tests for the QUESTS entropy backend and trajectory selection.

Covers the typed :class:`QuestsSplitConfig`, the :class:`FrameDescriptors`
slicing helper, the explicit CPU/GPU backend resolution in
:class:`QuestsObjective`, entropy-profile evaluation on any trajectory, and
the nested maximum-information-entropy trajectory generation. Real-branch
tests run against the installed ``quests==2026.2.22`` package on the CPU
route; the GPU route is exercised with explicit fakes at the ordinary backend
import boundary (no torch dependency, no Protocol/registry).
"""
from __future__ import annotations

import sys
import unittest
from typing import Sequence

import numpy as np
from ase import Atoms
from pydantic import ValidationError

from src.temper.schemas import (
    EntropyProfile,
    FrameReference,
    QuestsSplitConfig,
    TrainValSplitTrajectory,
)
from src.temper.splitting import (
    FrameDescriptors,
    QuestsNumericalError,
    QuestsObjective,
    QuestsUnavailableError,
    build_frame_descriptors,
    evaluate_entropy_profile,
    generate_quests_trajectory,
    populate_entropy_profile,
)
from src.temper.splitting.random import generate_random_trajectory


def make_reference(
    domain: str = "d",
    filename: str = "a.extxyz",
    frame_index: int = 0,
) -> FrameReference:
    """Build a :class:`FrameReference` with overridable fields."""
    return FrameReference(
        domain=domain,
        filename=filename,
        frame_index=frame_index,
    )


def make_pool(n_frames: int = 8) -> list[FrameReference]:
    """Build a pool of ``n_frames`` distinct references."""
    return [make_reference(frame_index=i) for i in range(n_frames)]


def make_structures(n_frames: int = 8) -> list[Atoms]:
    """Build a deterministic pool of non-PBC clustered structures.

    The frames carry 4-6 atoms each spread in a small box around a frame
    specific center, so real QUESTS descriptors/entropy stay finite with the
    test bandwidth (see ``test_config``).
    """
    rng = np.random.default_rng(42)
    frames: list[Atoms] = []
    for frame_index in range(n_frames):
        n_atoms = 4 + (frame_index % 3)
        center = np.array([frame_index * 0.4, 0.0, 0.0])
        positions = center + rng.uniform(-1.25, 1.25, size=(n_atoms, 3))
        frames.append(Atoms("C" * n_atoms, positions=positions))
    return frames


def cpu_config(**overrides: object) -> QuestsSplitConfig:
    """Build a CPU config with numerically safe parameters for the small frames."""
    defaults: dict[str, object] = {
        "descriptor_k": 4,
        "descriptor_cutoff": 5.0,
        "entropy_bandwidth": 1.0,
        "entropy_batch_size": 20000,
        "device": "cpu",
    }
    defaults.update(overrides)
    return QuestsSplitConfig(**defaults)  # type: ignore[arg-type]


class FakeObjective:
    """Deterministic fake :class:`QuestsObjective` for selection-logic tests.

    Each frame is a single atom whose scalar descriptor is the x-coordinate of
    its first atom. ``entropy`` is the descriptor sum and ``delta_entropy`` is
    the absolute difference to the reference mean, so the greedy selection
    behavior is fully predictable.
    """

    def compute_descriptors(self, structures: Sequence[Atoms]) -> FrameDescriptors:
        """Build one scalar descriptor per frame from the first atom x-coord."""
        rows: list[np.ndarray] = []
        offsets = [0]
        for atoms in structures:
            rows.append(np.array([[atoms.positions[0, 0]]], dtype=np.float64))
            offsets.append(offsets[-1] + 1)
        return FrameDescriptors(
            values=np.concatenate(rows, axis=0),
            frame_offsets=tuple(offsets),
        )

    def entropy(self, descriptors: np.ndarray) -> float:
        """Sum of the descriptor values."""
        return float(np.sum(np.asarray(descriptors)))

    def delta_entropy(self, candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Per-row absolute difference to the reference mean."""
        candidate_matrix = np.asarray(candidate, dtype=np.float64)
        reference_matrix = np.asarray(reference, dtype=np.float64)
        return np.abs(candidate_matrix[:, 0] - np.mean(reference_matrix[:, 0]))


class CountingObjective(FakeObjective):
    """Fake objective that counts descriptor computations.

    Delegates to :class:`FakeObjective` but records every
    ``compute_descriptors`` call, so tests can assert that pool descriptors
    are computed exactly once and reused across selection/profile evaluation.
    """

    def __init__(self) -> None:
        """Initialize the descriptor-computation counter."""
        self.descriptor_calls = 0

    def compute_descriptors(self, structures: Sequence[Atoms]) -> FrameDescriptors:
        """Count the call, then delegate to the fake base implementation."""
        self.descriptor_calls += 1
        return super().compute_descriptors(structures)


class TestObjectiveConfigConsistency(unittest.TestCase):
    """Tests for the objective/config consistency enforcement."""

    def test_build_frame_descriptors_rejects_mismatch(self) -> None:
        config = cpu_config()
        mismatched = QuestsObjective(cpu_config(entropy_bandwidth=2.0))
        with self.assertRaises(ValueError) as ctx:
            build_frame_descriptors(
                make_structures(),
                config=config,
                objective=mismatched,
            )
        self.assertIn("does not match", str(ctx.exception))

    def test_evaluate_entropy_profile_rejects_mismatch(self) -> None:
        pool = make_pool()
        structures = make_structures()
        trajectory = generate_random_trajectory(
            seed=1,
            pool=pool,
            requested_train_sizes=[2],
        )
        mismatched = QuestsObjective(cpu_config(entropy_bandwidth=2.0))
        with self.assertRaises(ValueError) as ctx:
            evaluate_entropy_profile(
                trajectory=trajectory,
                pool=pool,
                structures=structures,
                config=cpu_config(),
                objective=mismatched,
            )
        self.assertIn("does not match", str(ctx.exception))

    def test_populate_entropy_profile_rejects_mismatch(self) -> None:
        pool = make_pool()
        structures = make_structures()
        trajectory = generate_random_trajectory(
            seed=1,
            pool=pool,
            requested_train_sizes=[2],
        )
        mismatched = QuestsObjective(cpu_config(entropy_bandwidth=2.0))
        with self.assertRaises(ValueError) as ctx:
            populate_entropy_profile(
                trajectory=trajectory,
                pool=pool,
                structures=structures,
                config=cpu_config(),
                objective=mismatched,
            )
        self.assertIn("does not match", str(ctx.exception))

    def test_generate_quests_trajectory_rejects_mismatch(self) -> None:
        pool = make_pool()
        structures = make_structures()
        mismatched = QuestsObjective(cpu_config(entropy_bandwidth=2.0))
        with self.assertRaises(ValueError) as ctx:
            generate_quests_trajectory(
                pool=pool,
                structures=structures,
                requested_train_sizes=[2],
                config=cpu_config(),
                objective=mismatched,
            )
        self.assertIn("does not match", str(ctx.exception))

    def test_matching_config_is_accepted(self) -> None:
        config = cpu_config()
        objective = QuestsObjective(config)
        # No exception raised: a matching objective is fully supported.
        descriptors = build_frame_descriptors(
            make_structures(),
            config=config,
            objective=objective,
        )
        self.assertEqual(descriptors.n_frames, 8)

    def test_fake_objective_without_config_is_accepted(self) -> None:
        # Deterministic fakes expose no .config and must keep working.
        descriptors = build_frame_descriptors(
            make_structures(),
            config=cpu_config(),
            objective=FakeObjective(),
        )
        self.assertEqual(descriptors.n_frames, 8)


class TestPrecomputedDescriptorReuse(unittest.TestCase):
    """Regression tests for the shared precomputed-descriptor route."""

    def test_quests_trajectory_computes_descriptors_once(self) -> None:
        pool = make_pool()
        structures = make_structures()
        objective = CountingObjective()
        trajectory = generate_quests_trajectory(
            pool=pool,
            structures=structures,
            requested_train_sizes=[2, 4, 6],
            config=cpu_config(),
            objective=objective,
        )
        # Greedy selection and profile evaluation share one descriptor pass.
        self.assertEqual(objective.descriptor_calls, 1)
        self.assertIsNotNone(trajectory.entropy_profile)

    def test_quests_trajectory_reuses_precomputed_descriptors(self) -> None:
        pool = make_pool()
        structures = make_structures()
        config = cpu_config()
        objective = CountingObjective()
        descriptors = build_frame_descriptors(
            structures,
            config=config,
            objective=objective,
        )
        trajectory = generate_quests_trajectory(
            pool=pool,
            structures=structures,
            requested_train_sizes=[2, 4],
            config=config,
            objective=objective,
            descriptors=descriptors,
        )
        # No descriptor recomputation inside generate_quests_trajectory.
        self.assertEqual(objective.descriptor_calls, 1)
        reference = generate_quests_trajectory(
            pool=pool,
            structures=structures,
            requested_train_sizes=[2, 4],
            config=config,
            objective=FakeObjective(),
        )
        self.assertEqual(
            [ref.identity for ref in trajectory.selected_frames],
            [ref.identity for ref in reference.selected_frames],
        )
        assert trajectory.entropy_profile is not None
        assert reference.entropy_profile is not None
        self.assertEqual(trajectory.entropy_profile, reference.entropy_profile)

    def test_precomputed_pool_descriptors_match_computed_profile(self) -> None:
        pool = make_pool()
        structures = make_structures()
        config = cpu_config()
        trajectory = generate_random_trajectory(
            seed=3,
            pool=pool,
            requested_train_sizes=[2, 4, 6],
        )
        descriptors = build_frame_descriptors(structures, config=config)
        direct = evaluate_entropy_profile(
            trajectory=trajectory,
            pool=pool,
            structures=structures,
            config=config,
        )
        precomputed = evaluate_entropy_profile(
            trajectory=trajectory,
            pool=pool,
            structures=structures,
            config=config,
            descriptors=descriptors,
        )
        self.assertEqual(direct, precomputed)

    def test_precomputed_descriptors_must_match_pool_size(self) -> None:
        pool = make_pool(8)
        structures = make_structures(8)
        config = cpu_config()
        trajectory = generate_random_trajectory(
            seed=3,
            pool=pool,
            requested_train_sizes=[2],
        )
        small = build_frame_descriptors(make_structures(3), config=config)
        with self.assertRaises(ValueError):
            evaluate_entropy_profile(
                trajectory=trajectory,
                pool=pool,
                structures=structures,
                config=config,
                descriptors=small,
            )


class TestQuestsSplitConfig(unittest.TestCase):
    """Tests for :class:`QuestsSplitConfig` validation and defaults."""

    def test_defaults(self) -> None:
        config = QuestsSplitConfig()
        self.assertEqual(config.descriptor_k, 32)
        self.assertEqual(config.descriptor_cutoff, 5.0)
        self.assertEqual(config.descriptor_dtype, "float64")
        self.assertEqual(config.entropy_bandwidth, 0.015)
        self.assertEqual(config.entropy_batch_size, 20000)
        self.assertEqual(config.entropy_tolerance, 1e-6)
        self.assertEqual(config.device, "auto")
        self.assertIsNone(config.gpu_device)
        self.assertIsNone(config.numba_threads)
        # Entropy tolerance remains a configuration field for compatibility,
        # but profile validation does not impose a nonnegativity constraint.
        self.assertEqual(config.entropy_tolerance, 1e-6)

    def test_rejects_small_descriptor_k(self) -> None:
        with self.assertRaises(ValidationError):
            QuestsSplitConfig(descriptor_k=1)

    def test_rejects_non_positive_cutoff(self) -> None:
        with self.assertRaises(ValidationError):
            QuestsSplitConfig(descriptor_cutoff=0.0)

    def test_rejects_non_positive_bandwidth(self) -> None:
        with self.assertRaises(ValidationError):
            QuestsSplitConfig(entropy_bandwidth=0.0)

    def test_rejects_non_positive_batch_size(self) -> None:
        with self.assertRaises(ValidationError):
            QuestsSplitConfig(entropy_batch_size=0)

    def test_rejects_negative_tolerance(self) -> None:
        with self.assertRaises(ValidationError):
            QuestsSplitConfig(entropy_tolerance=-1.0)

    def test_rejects_non_positive_numba_threads(self) -> None:
        with self.assertRaises(ValidationError):
            QuestsSplitConfig(numba_threads=0)

    def test_rejects_invalid_device(self) -> None:
        with self.assertRaises(ValidationError):
            QuestsSplitConfig(device="quantum")

    def test_rejects_invalid_dtype(self) -> None:
        with self.assertRaises(ValidationError):
            QuestsSplitConfig(descriptor_dtype="float16")

    def test_rejects_gpu_device_on_cpu(self) -> None:
        with self.assertRaises(ValidationError):
            QuestsSplitConfig(device="cpu", gpu_device="cuda:0")

    def test_accepts_gpu_device_on_gpu_route(self) -> None:
        config = QuestsSplitConfig(device="gpu", gpu_device="cuda:1")
        self.assertEqual(config.gpu_device, "cuda:1")

    def test_accepts_gpu_device_on_auto_route(self) -> None:
        config = QuestsSplitConfig(device="auto", gpu_device="cuda:0")
        self.assertEqual(config.device, "auto")

    def test_round_trip(self) -> None:
        config = cpu_config()
        restored = QuestsSplitConfig.model_validate(config.model_dump())
        self.assertEqual(restored, config)


class _FakeGpuTensor:
    """Minimal torch-tensor stand-in for GPU adapter contract tests.

    Implements only the ``detach()``/``cpu()``/``item()``/``numpy()`` surface
    that :class:`QuestsObjective` consumes after a GPU backend call.
    """

    def __init__(self, value: object) -> None:
        """Store the wrapped scalar or array value."""
        self._value = value

    def detach(self) -> "_FakeGpuTensor":
        """Return self (no autograd graph exists)."""
        return self

    def cpu(self) -> "_FakeGpuTensor":
        """Return self (already on the host)."""
        return self

    def item(self) -> float:
        """Return the wrapped scalar as a Python float."""
        return float(self._value)

    def numpy(self) -> np.ndarray:
        """Return the wrapped value as a NumPy array."""
        return np.asarray(self._value)


class _FakeGpuEntropy:
    """Fake ``quests.gpu.entropy`` module used by GPU contract tests.

    The ``entropy``/``delta_entropy`` signatures mirror the real QUESTS GPU
    module (candidate/reference argument order plus a ``device`` string).
    """

    @staticmethod
    def entropy(x: object, h: object, batch_size: object, device: str) -> _FakeGpuTensor:
        """Return a fixed scalar entropy tensor, ignoring inputs."""
        del x, h, batch_size, device
        return _FakeGpuTensor(42.0)

    @staticmethod
    def delta_entropy(
        x: object,
        y: object,
        h: object,
        batch_size: object,
        device: str,
    ) -> _FakeGpuTensor:
        """Return a per-row constant delta-entropy tensor."""
        del y, h, batch_size, device
        n_rows = np.asarray(x).shape[0]
        return _FakeGpuTensor(np.full((n_rows,), 1.5))


class TestFrameDescriptors(unittest.TestCase):
    """Tests for :class:`FrameDescriptors` slicing and concatenation."""

    @staticmethod
    def make_descriptors() -> FrameDescriptors:
        """Build a 5-row, 3-frame descriptor set with known offsets."""
        values = np.arange(10, dtype=np.float64).reshape(5, 2)
        return FrameDescriptors(values=values, frame_offsets=(0, 2, 4, 5))

    def test_properties(self) -> None:
        descriptors = self.make_descriptors()
        self.assertEqual(descriptors.n_frames, 3)
        self.assertEqual(descriptors.n_dims, 2)
        self.assertEqual(descriptors.frame_atom_counts, (2, 2, 1))

    def test_slice(self) -> None:
        descriptors = self.make_descriptors()
        np.testing.assert_array_equal(
            descriptors.slice(1),
            np.array([[4.0, 5.0], [6.0, 7.0]]),
        )

    def test_concat(self) -> None:
        descriptors = self.make_descriptors()
        np.testing.assert_array_equal(
            descriptors.concat([2, 0]),
            np.array([[8.0, 9.0], [0.0, 1.0], [2.0, 3.0]]),
        )

    def test_rejects_empty_concat(self) -> None:
        with self.assertRaises(ValueError):
            self.make_descriptors().concat([])

    def test_rejects_out_of_range_index(self) -> None:
        with self.assertRaises(IndexError):
            self.make_descriptors().slice(3)

    def test_rejects_offset_not_starting_at_zero(self) -> None:
        with self.assertRaises(ValueError):
            FrameDescriptors(values=np.zeros((4, 2)), frame_offsets=(1, 2, 4))

    def test_rejects_non_increasing_offsets(self) -> None:
        with self.assertRaises(ValueError):
            FrameDescriptors(values=np.zeros((4, 2)), frame_offsets=(0, 3, 3))

    def test_rejects_offset_mismatching_rows(self) -> None:
        with self.assertRaises(ValueError):
            FrameDescriptors(values=np.zeros((4, 2)), frame_offsets=(0, 2, 3))

    def test_rejects_1d_values(self) -> None:
        with self.assertRaises(ValueError):
            FrameDescriptors(values=np.zeros(4), frame_offsets=(0, 4))


class TestQuestsObjectiveCpu(unittest.TestCase):
    """Real-backend CPU route tests for :class:`QuestsObjective`."""

    def test_resolve_cpu(self) -> None:
        objective = QuestsObjective(cpu_config())
        self.assertEqual(objective.resolve_device(), "cpu")

    def test_compute_descriptors_shape_and_slices(self) -> None:
        structures = make_structures()
        objective = QuestsObjective(cpu_config())
        descriptors = objective.compute_descriptors(structures)
        total_atoms = sum(len(atoms) for atoms in structures)
        self.assertEqual(descriptors.values.shape, (total_atoms, 2 * 4 - 1))
        self.assertEqual(
            descriptors.frame_atom_counts,
            tuple(len(atoms) for atoms in structures),
        )

    def test_entropy_and_delta_entropy_finite(self) -> None:
        structures = make_structures()
        objective = QuestsObjective(cpu_config())
        descriptors = objective.compute_descriptors(structures)
        full = descriptors.concat(list(range(descriptors.n_frames)))
        entropy = objective.entropy(full)
        self.assertTrue(np.isfinite(entropy))
        deltas = objective.delta_entropy(
            descriptors.slice(0),
            descriptors.concat([1, 2, 3]),
        )
        self.assertEqual(deltas.shape, (descriptors.frame_atom_counts[0],))
        self.assertTrue(np.all(np.isfinite(deltas)))

    def test_cpu_route_does_not_import_torch(self) -> None:
        # Ensure torch is not already imported, then exercise the CPU route.
        sys.modules.pop("torch", None)
        objective = QuestsObjective(cpu_config())
        descriptors = objective.compute_descriptors(make_structures())
        objective.entropy(descriptors.concat([0, 1]))
        objective.delta_entropy(descriptors.slice(0), descriptors.slice(1))
        self.assertNotIn("torch", sys.modules)

    def test_rejects_empty_structures(self) -> None:
        objective = QuestsObjective(cpu_config())
        with self.assertRaises(ValueError):
            objective.compute_descriptors([])

    def test_rejects_atomless_structure(self) -> None:
        objective = QuestsObjective(cpu_config())
        with self.assertRaises(ValueError):
            objective.compute_descriptors([Atoms()])

    def test_rejects_feature_dimension_mismatch(self) -> None:
        objective = QuestsObjective(cpu_config())
        with self.assertRaises(ValueError):
            objective.delta_entropy(np.ones((2, 3)), np.ones((4, 5)))

    def test_rejects_empty_candidate(self) -> None:
        objective = QuestsObjective(cpu_config())
        with self.assertRaises(ValueError):
            objective.delta_entropy(np.zeros((0, 3)), np.ones((4, 3)))


class TestQuestsObjectiveGpu(unittest.TestCase):
    """Device-resolution and fake-backend GPU route tests."""

    def test_gpu_request_without_cuda_raises_clearly(self) -> None:
        objective = QuestsObjective(cpu_config(device="gpu"))

        def no_gpu() -> None:
            raise QuestsUnavailableError("torch is not installed")

        objective._assert_gpu_available = no_gpu  # type: ignore[method-assign]
        with self.assertRaises(QuestsUnavailableError) as ctx:
            objective.resolve_device()
        self.assertIn("torch", str(ctx.exception))

    def test_auto_falls_back_to_cpu_when_gpu_unavailable(self) -> None:
        objective = QuestsObjective(cpu_config(device="auto"))

        def no_gpu() -> None:
            raise QuestsUnavailableError("no CUDA device")

        objective._assert_gpu_available = no_gpu  # type: ignore[method-assign]
        self.assertEqual(objective.resolve_device(), "cpu")

    def test_auto_uses_gpu_when_available(self) -> None:
        objective = QuestsObjective(cpu_config(device="auto"))

        def has_gpu() -> None:
            return None

        objective._assert_gpu_available = has_gpu  # type: ignore[method-assign]
        self.assertEqual(objective.resolve_device(), "gpu")

    def test_auto_route_does_not_hide_invalid_gpu_device(self) -> None:
        objective = QuestsObjective(
            cpu_config(device="auto", gpu_device="cuda:99")
        )

        def invalid_gpu_device() -> None:
            raise ValueError("CUDA device is invalid")

        objective._assert_gpu_available = invalid_gpu_device  # type: ignore[method-assign]
        with self.assertRaises(ValueError):
            objective.resolve_device()

    def test_real_gpu_assertion_when_torch_missing(self) -> None:
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            torch = None
        if torch is None:
            objective = QuestsObjective(cpu_config(device="gpu"))
            with self.assertRaises(QuestsUnavailableError):
                objective._assert_gpu_available()

    def test_gpu_entropy_and_delta_entropy_via_fake_backend(self) -> None:
        """Contract test: torch tensors in, float/array out, finite-checked."""
        objective = QuestsObjective(cpu_config(device="gpu"))
        calls: dict[str, int] = {"to_tensor": 0}

        def fake_to_tensor(array: np.ndarray) -> np.ndarray:
            calls["to_tensor"] += 1
            return np.asarray(array)

        objective._assert_gpu_available = lambda: None  # type: ignore[method-assign]
        objective._import_gpu_entropy = lambda: _FakeGpuEntropy()  # type: ignore[method-assign]
        objective._to_tensor = fake_to_tensor  # type: ignore[method-assign]

        entropy = objective.entropy(np.ones((4, 3)))
        self.assertEqual(entropy, 42.0)
        deltas = objective.delta_entropy(np.ones((2, 3)), np.ones((5, 3)))
        np.testing.assert_array_equal(deltas, np.full((2,), 1.5))
        # One conversion for entropy, two (candidate + reference) for delta.
        self.assertEqual(calls["to_tensor"], 3)

    def test_to_tensor_when_torch_available(self) -> None:
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            self.skipTest("torch is not installed in this environment")
        if not torch.cuda.is_available():
            self.skipTest("no CUDA device available")
        objective = QuestsObjective(cpu_config(device="gpu", gpu_device="cuda:0"))
        tensor = objective._to_tensor(np.ones((2, 3)))
        self.assertEqual(str(tensor.device).split(":")[0], "cuda")
        self.assertEqual(tensor.dtype, torch.float64)


class TestQuestsNumericalValidation(unittest.TestCase):
    """Fake-backend tests for non-finite output rejection."""

    def make_objective(self, entropy_result: object, delta_result: object) -> QuestsObjective:
        """Build a CPU objective with a fake backend returning fixed results.

        Parameters
        ----------
        entropy_result : object
            Value returned by the fake ``entropy`` backend.
        delta_result : object
            Value returned by the fake ``delta_entropy`` backend.

        Returns
        -------
        QuestsObjective
            An objective whose CPU backend is replaced by fakes.
        """
        objective = QuestsObjective(cpu_config())

        class FakeDescriptor:
            @staticmethod
            def get_descriptors(
                frames: Sequence[Atoms],
                k: int,
                cutoff: float,
                concat: bool,
                dtype: str,
            ) -> np.ndarray:
                del k, cutoff, concat, dtype
                rows = [np.full((len(atoms), 3), 1.0) for atoms in frames]
                return np.concatenate(rows, axis=0)

        class FakeEntropy:
            @staticmethod
            def entropy(x: object, h: object, batch_size: object) -> object:
                del h, batch_size
                return entropy_result

            @staticmethod
            def delta_entropy(y: object, x: object, h: object, batch_size: object) -> object:
                del h, batch_size
                return delta_result

        objective._import_cpu_backend = lambda: (  # type: ignore[method-assign]
            FakeDescriptor(),
            FakeEntropy(),
        )
        return objective

    def test_entropy_rejects_inf(self) -> None:
        objective = self.make_objective(float("inf"), np.zeros((2,)))
        with self.assertRaises(QuestsNumericalError):
            objective.entropy(np.ones((3, 3)))

    def test_delta_entropy_rejects_nan(self) -> None:
        objective = self.make_objective(1.0, np.array([1.0, np.nan]))
        with self.assertRaises(QuestsNumericalError):
            objective.delta_entropy(np.ones((2, 3)), np.ones((4, 3)))

    def test_delta_entropy_rejects_wrong_shape(self) -> None:
        objective = self.make_objective(1.0, np.zeros(5))
        with self.assertRaises(QuestsNumericalError):
            objective.delta_entropy(np.ones((2, 3)), np.ones((4, 3)))

    def test_descriptors_reject_nan(self) -> None:
        objective = QuestsObjective(cpu_config())

        class NanDescriptor:
            @staticmethod
            def get_descriptors(
                frames: Sequence[Atoms],
                k: int,
                cutoff: float,
                concat: bool,
                dtype: str,
            ) -> np.ndarray:
                del k, cutoff, concat, dtype
                return np.full((sum(len(a) for a in frames), 3), np.nan)

        objective._import_cpu_backend = lambda: (  # type: ignore[method-assign]
            NanDescriptor(),
            None,
        )
        with self.assertRaises(QuestsNumericalError):
            objective.compute_descriptors(make_structures())


class TestEvaluateEntropyProfile(unittest.TestCase):
    """Real-backend tests for entropy-profile evaluation on any trajectory."""

    def test_populates_random_trajectory_profile(self) -> None:
        pool = make_pool()
        structures = make_structures()
        config = cpu_config()
        trajectory = generate_random_trajectory(
            seed=3,
            pool=pool,
            requested_train_sizes=[2, 4, 6],
        )
        populated = populate_entropy_profile(
            trajectory=trajectory,
            pool=pool,
            structures=structures,
            config=config,
        )
        self.assertEqual(populated.method, "random")
        self.assertIsNotNone(populated.entropy_profile)
        profile = populated.entropy_profile
        assert profile is not None
        self.assertEqual(
            [point.training_size for point in profile.points],
            [2, 4, 6],
        )
        for point in profile.points:
            self.assertTrue(np.isfinite(point.cumulative_entropy))
            self.assertTrue(np.isfinite(point.information_gain))
        # The first chunk has no reference set, so its gain equals its entropy.
        self.assertAlmostEqual(
            profile.points[0].information_gain,
            profile.points[0].cumulative_entropy,
            places=6,
        )
        # Original trajectory is unchanged.
        self.assertIsNone(trajectory.entropy_profile)

    def test_evaluate_matches_populate(self) -> None:
        pool = make_pool()
        structures = make_structures()
        config = cpu_config()
        trajectory = generate_random_trajectory(
            seed=5,
            pool=pool,
            requested_train_sizes=[1, 3, 5],
        )
        direct = evaluate_entropy_profile(
            trajectory=trajectory,
            pool=pool,
            structures=structures,
            config=config,
        )
        populated = populate_entropy_profile(
            trajectory=trajectory,
            pool=pool,
            structures=structures,
            config=config,
        )
        assert populated.entropy_profile is not None
        self.assertEqual(direct, populated.entropy_profile)

    def test_rejects_selected_frame_outside_pool(self) -> None:
        pool = make_pool(4)
        structures = make_structures(4)
        outside = TrainValSplitTrajectory(
            method="random",
            seed=1,
            requested_train_sizes=[1],
            selected_frames=[make_reference(frame_index=99)],
        )
        with self.assertRaises(ValueError):
            evaluate_entropy_profile(
                trajectory=outside,
                pool=pool,
                structures=structures,
                config=cpu_config(),
            )


class TestGenerateQuestsTrajectory(unittest.TestCase):
    """Real-backend tests for nested maximum-information-entropy selection."""

    def test_deterministic_and_nested(self) -> None:
        pool = make_pool()
        structures = make_structures()
        config = cpu_config()
        first = generate_quests_trajectory(
            pool=pool,
            structures=structures,
            requested_train_sizes=[2, 4, 6],
            config=config,
        )
        second = generate_quests_trajectory(
            pool=pool,
            structures=structures,
            requested_train_sizes=[2, 4, 6],
            config=config,
        )
        self.assertEqual(first.method, "quests")
        self.assertIsNone(first.seed)
        self.assertEqual(
            [ref.identity for ref in first.selected_frames],
            [ref.identity for ref in second.selected_frames],
        )
        self.assertEqual(len(first.selected_frames), 6)
        selected_identities = [ref.identity for ref in first.selected_frames]
        self.assertEqual(len(set(selected_identities)), 6)
        self.assertTrue(set(selected_identities) <= {ref.identity for ref in pool})

        train_2 = {ref.identity for ref in first.get_train_set(0)}
        train_4 = {ref.identity for ref in first.get_train_set(1)}
        train_6 = {ref.identity for ref in first.get_train_set(2)}
        self.assertTrue(train_2 <= train_4 <= train_6)

        profile = first.entropy_profile
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(
            [point.training_size for point in profile.points],
            [2, 4, 6],
        )
        # Snapshot access remains restricted to requested sizes.
        self.assertEqual(len(first.get_train_set(0)), 2)
        self.assertEqual(
            {ref.identity for ref in first.selected_frames}
            | {ref.identity for ref in first.additional_trainval_frames},
            {ref.identity for ref in pool},
        )

    def test_profile_valid_and_finite(self) -> None:
        pool = make_pool()
        structures = make_structures()
        trajectory = generate_quests_trajectory(
            pool=pool,
            structures=structures,
            requested_train_sizes=[2, 4, 6],
            config=cpu_config(),
        )
        profile = trajectory.entropy_profile
        assert profile is not None
        for point in profile.points:
            self.assertTrue(np.isfinite(point.cumulative_entropy))
            self.assertTrue(np.isfinite(point.information_gain))

    def test_full_pool_selection(self) -> None:
        pool = make_pool()
        structures = make_structures()
        trajectory = generate_quests_trajectory(
            pool=pool,
            structures=structures,
            requested_train_sizes=[8],
            config=cpu_config(),
        )
        self.assertEqual(
            {ref.identity for ref in trajectory.selected_frames},
            {ref.identity for ref in pool},
        )


class TestGreedySelectionLogic(unittest.TestCase):
    """Deterministic fake-objective tests for the selection algorithm."""

    def test_selects_maximum_self_entropy_first(self) -> None:
        # Single-atom frames with descriptor values 0, 10, 20 -> max first.
        structures = [
            Atoms("C", positions=[[0.0, 0.0, 0.0]]),
            Atoms("C", positions=[[10.0, 0.0, 0.0]]),
            Atoms("C", positions=[[20.0, 0.0, 0.0]]),
        ]
        pool = make_pool(3)
        trajectory = generate_quests_trajectory(
            pool=pool,
            structures=structures,
            requested_train_sizes=[3],
            config=cpu_config(),
            objective=FakeObjective(),
        )
        self.assertEqual(
            [ref.frame_index for ref in trajectory.selected_frames],
            [2, 0, 1],
        )

    def test_selection_conditioned_on_selected_descriptors(self) -> None:
        # Values [0, 10, 20]: pick 20, then the farthest from it (0), then the
        # remaining one (10) — each step conditioned on the selected set.
        structures = [
            Atoms("C", positions=[[0.0, 0.0, 0.0]]),
            Atoms("C", positions=[[10.0, 0.0, 0.0]]),
            Atoms("C", positions=[[20.0, 0.0, 0.0]]),
        ]
        pool = make_pool(3)
        trajectory = generate_quests_trajectory(
            pool=pool,
            structures=structures,
            requested_train_sizes=[1, 2, 3],
            config=cpu_config(),
            objective=FakeObjective(),
        )
        profile = trajectory.entropy_profile
        assert profile is not None
        self.assertEqual(
            [point.training_size for point in profile.points],
            [1, 2, 3],
        )
        self.assertEqual(
            [point.cumulative_entropy for point in profile.points],
            [20.0, 20.0, 30.0],
        )
        # First chunk gain = its own entropy; later chunks = delta sums.
        self.assertEqual(
            [point.information_gain for point in profile.points],
            [20.0, 20.0, 0.0],
        )

    def test_multi_frame_selection_step_has_one_profile_point(self) -> None:
        structures = [
            Atoms("C", positions=[[0.0, 0.0, 0.0]]),
            Atoms("C", positions=[[10.0, 0.0, 0.0]]),
            Atoms("C", positions=[[20.0, 0.0, 0.0]]),
            Atoms("C", positions=[[30.0, 0.0, 0.0]]),
        ]
        pool = make_pool(4)
        trajectory = generate_random_trajectory(
            seed=1,
            pool=pool,
            requested_train_sizes=[2, 4],
        )
        profile = evaluate_entropy_profile(
            trajectory=trajectory,
            pool=pool,
            structures=structures,
            config=cpu_config(),
            objective=FakeObjective(),
        )

        self.assertEqual([point.training_size for point in profile.points], [2, 4])
        self.assertEqual(
            [point.cumulative_entropy for point in profile.points],
            [10.0, 60.0],
        )
        self.assertEqual(len(profile.points), 2)

    def test_identical_candidates_tie_break_by_pool_order(self) -> None:
        # All descriptor values identical (0) -> every score ties, so pool
        # order wins deterministically.
        structures = [
            Atoms("C", positions=[[0.0, 0.0, 0.0]]),
            Atoms("C", positions=[[0.0, 1.0, 0.0]]),
            Atoms("C", positions=[[0.0, 0.0, 1.0]]),
        ]
        pool = make_pool(3)
        trajectory = generate_quests_trajectory(
            pool=pool,
            structures=structures,
            requested_train_sizes=[3],
            config=cpu_config(),
            objective=FakeObjective(),
        )
        self.assertEqual(
            [ref.frame_index for ref in trajectory.selected_frames],
            [0, 1, 2],
        )


class TestQuestTrajectoryEdgeCases(unittest.TestCase):
    """Edge-case validation for the QUESTS trajectory functions."""

    def test_empty_pool_raises(self) -> None:
        with self.assertRaises(ValueError):
            generate_quests_trajectory(
                pool=[],
                structures=[],
                requested_train_sizes=[1],
                config=cpu_config(),
            )

    def test_misaligned_pool_and_structures_raises(self) -> None:
        pool = make_pool(4)
        structures = make_structures(3)
        with self.assertRaises(ValueError):
            generate_quests_trajectory(
                pool=pool,
                structures=structures,
                requested_train_sizes=[1],
                config=cpu_config(),
            )

    def test_size_exceeding_pool_raises(self) -> None:
        pool = make_pool(3)
        structures = make_structures(3)
        with self.assertRaises(ValueError):
            generate_quests_trajectory(
                pool=pool,
                structures=structures,
                requested_train_sizes=[4],
                config=cpu_config(),
            )

    def test_misaligned_pool_and_structures_in_profile_raises(self) -> None:
        pool = make_pool(3)
        structures = make_structures(3)
        trajectory = generate_random_trajectory(
            seed=1,
            pool=pool,
            requested_train_sizes=[2],
        )
        with self.assertRaises(ValueError):
            evaluate_entropy_profile(
                trajectory=trajectory,
                pool=make_pool(4),
                structures=structures,
                config=cpu_config(),
            )


if __name__ == "__main__":
    unittest.main()
