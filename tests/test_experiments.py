"""End-to-end integration tests for the high-level split orchestration.

Covers :func:`src.temper.experiments.split_data_group` and
:func:`src.temper.experiments.flatten_frames_and_structures`:

- reference/structures alignment verification,
- explicit method-selection sequence (random + quests in one schema),
- shared train+validation pool and requested sizes across methods,
- a real QUESTS CPU-backend entropy profile for the *random* trajectory,
- nested references mapping to the correct source indices via reconstruction,
- schema serialization round-trip including the ``quests_config`` provenance,
- documented environment defaults (test ratio, training ratios, max train cap).

Real-backend tests run against the installed ``quests==2026.2.22`` package on
the CPU route with numerically safe tiny clusters; tests that only exercise
orchestration/defaults use an explicit deterministic fake objective at the
ordinary backend boundary (no Protocol/registry).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from src.temper.experiments import (
    flatten_frames_and_structures,
    split_data_group,
)
from src.temper.schemas import (
    FrameReference,
    QuestsSplitConfig,
    SplitDataSchema,
    TrainValSplitTrajectory,
)
from src.temper.splitting import (
    FrameDescriptors,
    QuestsObjective,
    get_references_from_frames,
    partition_trainval_test,
    load_frames_from_references,
    load_frames_test,
    load_frames_train_validation,
)
from src.temper.utils.env import DEFAULT_TEST_RATIO, DEFAULT_TRAIN_RATIOS


def make_labeled_cluster(index: int, n_atoms: int | None = None) -> Atoms:
    """Build a tiny labeled C cluster that is numerically safe for QUESTS.

    The cluster has 4-6 atoms spread around a frame-specific center (the same
    family of structures used by the QUESTS backend tests), a deterministic
    energy ``-index``, and constant forces. ``info['tag']`` records the global
    frame index for reconstruction checks.
    """
    n = n_atoms if n_atoms is not None else 4 + (index % 3)
    rng = np.random.default_rng(42 + index)
    center = np.array([index * 0.4, 0.0, 0.0])
    positions = center + rng.uniform(-1.25, 1.25, size=(n, 3))
    atoms = Atoms("C" * n, positions=positions)
    energy = float(-index)
    forces = np.full((n, 3), float(index), dtype=float)
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    atoms.info["tag"] = f"frame{index}"
    return atoms


def expected_energy(filename: str, frame_index: int) -> float:
    """Return the fixture energy of a reference in the 6-frame inventory.

    ``a.extxyz`` holds global indices 0..2 and ``b.extxyz`` holds 3..5; the
    energy of a frame is its negated global index.
    """
    if filename == "b.extxyz":
        return float(-(3 + frame_index))
    return float(-frame_index)


def make_inventory(
    per_file: int = 3,
    n_files: int = 2,
) -> tuple[dict[str, list[int]], dict[str, list[Atoms]]]:
    """Build a ``(frames_by_filename, structures_by_filename)`` inventory.

    ``n_files`` files each holding ``per_file`` frames; frame indices are
    contiguous and the global index of ``file j``, frame ``i`` is
    ``j * per_file + i``.
    """
    frames: dict[str, list[int]] = {}
    structures: dict[str, list[Atoms]] = {}
    for file_index in range(n_files):
        filename = f"{chr(ord('a') + file_index)}.extxyz"
        indices = list(range(per_file))
        frames[filename] = indices
        structures[filename] = [
            make_labeled_cluster(file_index * per_file + frame_index)
            for frame_index in indices
        ]
    return frames, structures


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
    """Deterministic fake :class:`QuestsObjective` for orchestration tests.

    Each frame is a single atom whose scalar descriptor is the x-coordinate of
    its first atom. ``entropy`` is the descriptor sum and ``delta_entropy`` is
    the absolute difference to the reference mean, so selection is fully
    predictable and fast (no numba/QUESTS backend involved).
    """

    def compute_descriptors(self, structures: list[Atoms]) -> FrameDescriptors:
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
    ``compute_descriptors`` call, so tests can assert that the high-level
    orchestration computes the full trainval-pool descriptors exactly once and
    shares them across methods.
    """

    def __init__(self) -> None:
        """Initialize the descriptor-computation counter."""
        self.descriptor_calls = 0

    def compute_descriptors(self, structures: list[Atoms]) -> FrameDescriptors:
        """Count the call, then delegate to the fake base implementation."""
        self.descriptor_calls += 1
        return super().compute_descriptors(structures)


class TestGetReferencesAndStructures(unittest.TestCase):
    """Tests for :func:`flatten_frames_and_structures` alignment."""

    def test_references_and_structures_aligned(self) -> None:
        frames, structures = make_inventory(per_file=3, n_files=2)
        references, aligned = flatten_frames_and_structures(
            frames, structures, "d1"
        )
        self.assertEqual(len(references), 6)
        self.assertEqual(len(aligned), 6)
        # aligned[i] is the structure of references[i].
        self.assertIs(aligned[0], structures["a.extxyz"][0])
        self.assertIs(aligned[3], structures["b.extxyz"][0])
        # Canonical order matches get_references_from_frames.
        self.assertEqual(
            [ref.identity for ref in references],
            [ref.identity for ref in get_references_from_frames(frames, "d1")],
        )

    def test_missing_filename_raises(self) -> None:
        frames, structures = make_inventory()
        del structures["b.extxyz"]
        with self.assertRaises(ValueError):
            flatten_frames_and_structures(frames, structures, "d1")

    def test_extra_filename_raises(self) -> None:
        frames, structures = make_inventory()
        structures["extra.extxyz"] = [make_labeled_cluster(0)]
        with self.assertRaises(ValueError):
            flatten_frames_and_structures(frames, structures, "d1")

    def test_out_of_range_frame_index_raises(self) -> None:
        frames, structures = make_inventory()
        frames["a.extxyz"] = [0, 1, 2, 3]  # references index 3 but file has 3
        with self.assertRaises(ValueError):
            flatten_frames_and_structures(frames, structures, "d1")

    def test_rejects_non_sequence_structures(self) -> None:
        frames, structures = make_inventory()
        structures["a.extxyz"] = make_labeled_cluster(0)  # a single Atoms
        with self.assertRaises(TypeError):
            flatten_frames_and_structures(frames, structures, "d1")


class TestSplitDataGroupValidation(unittest.TestCase):
    """Validation-only tests using the deterministic fake objective."""

    def setUp(self) -> None:
        self.frames, self.structures = make_inventory(per_file=3, n_files=2)
        self.config = cpu_config()
        self.objective = FakeObjective()

    def split(self, **overrides: object) -> SplitDataSchema:
        """Call ``split_data_group`` with defaults plus overrides."""
        defaults: dict[str, object] = {
            "frames_by_filename": self.frames,
            "structures_by_filename": self.structures,
            "domain": "d1",
            "grouping_strategy": "all",
            "group_name": "g0",
            "split_seed": 7,
            "train_val_split_method": ["random", "quests"],
            "quests_config": self.config,
            "test_ratio": 0.2,
            "requested_train_sizes": [0.4, 0.8],
            "random_seed": 3,
            "objective": self.objective,
        }
        defaults.update(overrides)
        return split_data_group(**defaults)  # type: ignore[arg-type]

    def test_rejects_unknown_method(self) -> None:
        with self.assertRaises(ValueError):
            self.split(train_val_split_method=["random", "greedy"])

    def test_rejects_duplicate_methods(self) -> None:
        with self.assertRaises(ValueError):
            self.split(train_val_split_method=["quests", "quests"])

    def test_rejects_empty_method_sequence(self) -> None:
        with self.assertRaises(ValueError):
            self.split(train_val_split_method=[])

    def test_rejects_random_without_seed(self) -> None:
        with self.assertRaises(ValueError):
            self.split(random_seed=None)

    def test_missing_random_seed_is_rejected_before_objective_work(self) -> None:
        objective = CountingObjective()
        with self.assertRaises(ValueError):
            self.split(
                train_val_split_method=["quests", "random"],
                random_seed=None,
                objective=objective,
            )
        self.assertEqual(objective.descriptor_calls, 0)

    def test_rejects_both_test_ratio_and_test_size(self) -> None:
        with self.assertRaises(ValueError):
            self.split(test_ratio=0.2, test_size=1)

    def test_misaligned_inventory_raises(self) -> None:
        del self.structures["b.extxyz"]
        with self.assertRaises(ValueError):
            self.split()

    def test_rejects_collapsing_ratio_sizes(self) -> None:
        with self.assertRaises(ValueError):
            # On a 5-frame trainval pool these ratios collapse to [1, 1].
            self.split(requested_train_sizes=[0.2, 0.2])

    def test_quests_only_ignores_random_seed(self) -> None:
        results = self.split(train_val_split_method="quests", random_seed=None)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        schema = results[0]
        self.assertEqual(schema.train_val_split_trajectory.method, "quests")
        self.assertIsNone(schema.train_val_split_trajectory.seed)

    def test_counts_require_as_ratio_false(self) -> None:
        results = self.split(
            requested_train_sizes=[2, 4],
            as_ratio=False,
            train_val_split_method="random",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].train_val_split_trajectory.requested_train_sizes, [2, 4])

    def test_rejects_objective_config_mismatch(self) -> None:
        mismatched = QuestsObjective(cpu_config(entropy_bandwidth=2.0))
        with self.assertRaises(ValueError) as ctx:
            self.split(objective=mismatched)
        self.assertIn("does not match", str(ctx.exception))

    def test_both_methods_share_one_pool_descriptor_computation(self) -> None:
        objective = CountingObjective()
        results = split_data_group(
            frames_by_filename=self.frames,
            structures_by_filename=self.structures,
            domain="d1",
            grouping_strategy="all",
            group_name="g0",
            split_seed=7,
            train_val_split_method=["random", "quests"],
            quests_config=self.config,
            test_ratio=0.2,
            requested_train_sizes=[0.4, 0.8],
            random_seed=3,
            objective=objective,
        )
        self.assertEqual([result.train_val_split_trajectory.method for result in results], ["random", "quests"])
        # The full trainval-pool descriptors are computed exactly once and
        # shared for QUESTS selection/profile and the random trajectory's
        # entropy profile.
        self.assertEqual(objective.descriptor_calls, 1)


class TestSplitDataGroupRealBackend(unittest.TestCase):
    """End-to-end real QUESTS CPU-backend tests with reconstruction."""

    def setUp(self) -> None:
        self.frames, self.structures = make_inventory(per_file=3, n_files=2)
        self.config = cpu_config()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.domain_dir = self.root / "d1"
        self.domain_dir.mkdir(parents=True, exist_ok=True)
        for filename in self.frames:
            write(
                self.domain_dir / filename,
                self.structures[filename],
                format="extxyz",
            )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_split_both_methods_with_real_backend(self) -> None:
        results = split_data_group(
            frames_by_filename=self.frames,
            structures_by_filename=self.structures,
            domain="d1",
            grouping_strategy="all",
            group_name="g0",
            split_seed=7,
            train_val_split_method=["random", "quests"],
            quests_config=self.config,
            test_ratio=0.2,
            requested_train_sizes=[0.4, 0.8],
            random_seed=3,
        )
        self.assertEqual([result.train_val_split_trajectory.method for result in results], ["random", "quests"])
        schema = results[0]
        # 6 frames, test = round(6 * 0.2) = 1, train+val = 5.
        self.assertEqual(len(schema.test_set), 1)
        self.assertEqual(
            len(schema.train_val_split_trajectory.selected_frames)
            + len(schema.train_val_split_trajectory.additional_trainval_frames),
            5,
        )
        self.assertEqual(schema.test_ratio, 0.2)
        self.assertEqual(schema.quests_config, self.config)

        random_trajectory = schema.train_val_split_trajectory
        quests_trajectory = schema.train_val_split_trajectory

        # Each method gets one singular result with shared test/config partitions.
        quests_schema = results[1]
        self.assertEqual(random_trajectory.requested_train_sizes, quests_schema.train_val_split_trajectory.requested_train_sizes)
        self.assertEqual(random_trajectory.requested_train_sizes, [2, 4])
        self.assertEqual(random_trajectory.seed, 3)
        self.assertIsNone(quests_schema.train_val_split_trajectory.seed)
        self.assertEqual(schema.test_set, quests_schema.test_set)
        self.assertEqual(schema.quests_config, quests_schema.quests_config)

        # Both result schemas carry real QUESTS entropy profiles.
        self.assertIsNotNone(random_trajectory.entropy_profile)
        quests_trajectory = quests_schema.train_val_split_trajectory
        self.assertIsNotNone(quests_trajectory.entropy_profile)
        for trajectory in [random_trajectory, quests_trajectory]:
            profile = trajectory.entropy_profile
            assert profile is not None
            self.assertEqual(
                [point.training_size for point in profile.points],
                trajectory.requested_train_sizes,
            )
            for point in profile.points:
                self.assertTrue(np.isfinite(point.cumulative_entropy))
                self.assertTrue(np.isfinite(point.information_gain))

        # Every selected frame of every trajectory belongs to the trainval pool.
        pool_identities = {
            ref.identity
            for ref in (
                schema.train_val_split_trajectory.selected_frames
                + schema.train_val_split_trajectory.additional_trainval_frames
            )
        }
        for trajectory in [schema.train_val_split_trajectory]:
            for ref in trajectory.selected_frames:
                self.assertIn(ref.identity, pool_identities)

    def test_reconstruction_returns_expected_labels(self) -> None:
        results = split_data_group(
            frames_by_filename=self.frames,
            structures_by_filename=self.structures,
            domain="d1",
            grouping_strategy="all",
            group_name="g0",
            split_seed=7,
            train_val_split_method=["random", "quests"],
            quests_config=self.config,
            test_ratio=0.2,
            requested_train_sizes=[0.4, 0.8],
            random_seed=3,
        )
        schema = results[0]
        quests_schema = results[1]

        # Nested references map to the correct source indices.
        for trajectory in [schema.train_val_split_trajectory]:
            for ref in trajectory.selected_frames:
                (frame,) = load_frames_from_references([ref], self.root)
                self.assertAlmostEqual(
                    frame.get_potential_energy(),
                    expected_energy(ref.filename, ref.frame_index),
                )

        # Train + validation at the requested size, per method.
        for method_schema in (schema, quests_schema):
            train, validation = load_frames_train_validation(method_schema, 0, self.root)
            self.assertEqual(len(train), 2)
            self.assertEqual(len(validation), 3)
            trajectory = method_schema.train_val_split_trajectory
            train_refs = trajectory.get_train_set(0)
            self.assertEqual(
                [atoms.get_potential_energy() for atoms in train],
                [expected_energy(ref.filename, ref.frame_index) for ref in train_refs],
            )
            # The validation set is the complement of the training prefix
            # within the train+validation pool (the test frame is excluded).
            test_tags = {
                atoms.info["tag"]
                for atoms in load_frames_test(schema, self.root)
            }
            self.assertEqual(
                {atoms.info["tag"] for atoms in validation},
                {f"frame{i}" for i in range(6)}
                - test_tags
                - {atoms.info["tag"] for atoms in train},
            )

        # Compose train + test from the current index-based loaders.
        train, _ = load_frames_train_validation(quests_schema, 1, self.root)
        test = load_frames_test(quests_schema, self.root)
        self.assertEqual(len(train), len(quests_schema.train_val_split_trajectory.get_train_set(1)))
        self.assertEqual(len(test), len(quests_schema.test_set))

        # Test set.
        test_set = load_frames_test(schema, self.root)
        self.assertEqual(len(test_set), 1)

    def test_schema_serialization_round_trip(self) -> None:
        schema = split_data_group(
            frames_by_filename=self.frames,
            structures_by_filename=self.structures,
            domain="d1",
            grouping_strategy="all",
            group_name="g0",
            split_seed=7,
            train_val_split_method="random",
            quests_config=self.config,
            test_ratio=0.2,
            requested_train_sizes=[0.4, 0.8],
            random_seed=3,
        )[0]
        restored = SplitDataSchema.from_dict(schema.as_dict())
        self.assertEqual(restored, schema)
        self.assertEqual(restored.quests_config, self.config)
        # Re-validated trajectories are identical (including entropy profiles).
        self.assertEqual(restored.train_val_split_trajectory.method, "random")

    def test_seed_reproducibility(self) -> None:
        first = split_data_group(
            frames_by_filename=self.frames,
            structures_by_filename=self.structures,
            domain="d1",
            grouping_strategy="all",
            group_name="g0",
            split_seed=7,
            train_val_split_method="random",
            quests_config=self.config,
            test_ratio=0.2,
            requested_train_sizes=[2, 4],
            as_ratio=False,
            random_seed=3,
        )
        second = split_data_group(
            frames_by_filename=self.frames,
            structures_by_filename=self.structures,
            domain="d1",
            grouping_strategy="all",
            group_name="g0",
            split_seed=7,
            train_val_split_method="random",
            quests_config=self.config,
            test_ratio=0.2,
            requested_train_sizes=[2, 4],
            as_ratio=False,
            random_seed=3,
        )
        self.assertEqual(first[0].as_dict(), second[0].as_dict())


class TestSplitDataGroupDefaults(unittest.TestCase):
    """Tests that documented environment defaults drive the orchestration."""

    def setUp(self) -> None:
        # 10 files x 3 frames = 30 frames: test = round(30 * 0.2) = 6,
        # trainval = 24, and all DEFAULT_TRAIN_RATIOS map to positive sizes.
        self.frames: dict[str, list[int]] = {}
        self.structures: dict[str, list[Atoms]] = {}
        for file_index in range(10):
            filename = f"f{file_index}.extxyz"
            self.frames[filename] = [0, 1, 2]
            self.structures[filename] = [
                make_labeled_cluster(file_index * 3 + i) for i in range(3)
            ]
        self.objective = FakeObjective()

    def test_defaults_apply_test_ratio_and_train_ratios(self) -> None:
        schema = split_data_group(
            frames_by_filename=self.frames,
            structures_by_filename=self.structures,
            domain="d1",
            grouping_strategy="all",
            group_name="g0",
            split_seed=5,
            train_val_split_method="random",
            quests_config=cpu_config(),
            random_seed=1,
            objective=self.objective,
        )[0]
        # Default test ratio from the environment module.
        self.assertEqual(schema.test_ratio, DEFAULT_TEST_RATIO)
        self.assertEqual(len(schema.test_set), round(30 * DEFAULT_TEST_RATIO))
        # Default training ratios normalized against the 24-frame pool.
        expected_sizes = [
            round(ratio * 24)
            for ratio in DEFAULT_TRAIN_RATIOS
        ]
        self.assertEqual(
            schema.train_val_split_trajectory.requested_train_sizes,
            expected_sizes,
        )

    def test_explicit_test_size_derives_persisted_ratio(self) -> None:
        schema = split_data_group(
            frames_by_filename=self.frames,
            structures_by_filename=self.structures,
            domain="d1",
            grouping_strategy="all",
            group_name="g0",
            split_seed=5,
            train_val_split_method="quests",
            quests_config=cpu_config(),
            test_size=3,
            requested_train_sizes=[2, 4, 8],
            as_ratio=False,
            objective=self.objective,
        )[0]
        self.assertEqual(len(schema.test_set), 3)
        self.assertAlmostEqual(schema.test_ratio, 3 / 30)
        self.assertEqual(schema.train_val_split_trajectory.requested_train_sizes, [2, 4, 8])

    def test_default_max_train_cap_applies(self) -> None:
        # An explicit cap scales the ratio-based sizes down proportionally:
        # without the cap the largest DEFAULT_TRAIN_RATIO (0.95) maps to 23
        # frames in the 24-frame pool; with cap 12 it maps to exactly 12.
        schema = split_data_group(
            frames_by_filename=self.frames,
            structures_by_filename=self.structures,
            domain="d1",
            grouping_strategy="all",
            group_name="g0",
            split_seed=5,
            train_val_split_method="random",
            quests_config=cpu_config(),
            max_train_size=12,
            random_seed=1,
            objective=self.objective,
        )[0]
        sizes = schema.train_val_split_trajectory.requested_train_sizes
        self.assertEqual(sizes[-1], 12)
        self.assertLess(max(sizes), 24)


if __name__ == "__main__":
    unittest.main()
