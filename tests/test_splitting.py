"""Focused tests for the shared and random dataset-splitting logic."""
from __future__ import annotations

import unittest

from src.temper.schemas import (
    FrameReference,
    SplitDataSchema,
    TrainValSplitTrajectory,
)
from src.temper.splitting import (
    get_references_from_frames,
    generate_random_trajectory,
    get_requested_train_sizes_from_ratios,
    partition_trainval_test,
)


def make_pool(n_frames: int = 10) -> list[FrameReference]:
    """Build a canonical pool of ``n_frames`` references in two files."""
    half = n_frames // 2
    frames_by_filename = {
        "a.extxyz": list(range(half)),
        "b.extxyz": list(range(half, n_frames)),
    }
    return get_references_from_frames(frames_by_filename, "d")


class TestGetReferencesFromFrames(unittest.TestCase):
    """Tests for :func:`get_references_from_frames`."""

    def test_deterministic_canonical_order(self) -> None:
        expected = get_references_from_frames(
            {"b.extxyz": [0, 1], "a.extxyz": [2, 3]},
            "d",
        )
        actual = get_references_from_frames(
            {"a.extxyz": [3, 2], "b.extxyz": [1, 0]},
            "d",
        )
        self.assertEqual(
            [ref.identity for ref in actual],
            [ref.identity for ref in expected],
        )
        # Filenames sorted, then frame indices sorted.
        self.assertEqual(actual[0].filename, "a.extxyz")
        self.assertEqual(actual[0].frame_index, 2)
        self.assertEqual(actual[1].frame_index, 3)
        self.assertEqual(actual[2].filename, "b.extxyz")
        self.assertEqual(actual[2].frame_index, 0)

    def test_rejects_duplicate_frame_indices_in_file(self) -> None:
        with self.assertRaises(ValueError):
            get_references_from_frames({"a.extxyz": [0, 0]}, "d")

class TestPartitionTrainvalTest(unittest.TestCase):
    """Tests for :func:`partition_trainval_test`."""

    def test_original_pool_positions_are_used_for_both_outputs(self) -> None:
        pool = make_pool(10)
        trainval, test = partition_trainval_test(pool, seed=1, test_ratio=0.4)
        self.assertEqual(len(trainval), 6)
        self.assertEqual(len(test), 4)
        self.assertEqual(
            {ref.identity for ref in trainval} | {ref.identity for ref in test},
            {ref.identity for ref in pool},
        )
        self.assertTrue(
            {ref.identity for ref in trainval}.isdisjoint(
                {ref.identity for ref in test}
            )
        )

    def test_seed_reproducibility(self) -> None:
        pool = make_pool(50)
        first = partition_trainval_test(pool, seed=7, test_ratio=0.2)
        second = partition_trainval_test(pool, seed=7, test_ratio=0.2)
        self.assertEqual(
            [ref.identity for ref in first[0]],
            [ref.identity for ref in second[0]],
        )
        self.assertEqual(
            [ref.identity for ref in first[1]],
            [ref.identity for ref in second[1]],
        )

    def test_partition_reconstructs_pool(self) -> None:
        pool = make_pool(50)
        trainval, test = partition_trainval_test(
            pool, seed=7, test_ratio=0.2
        )
        combined = {ref.identity for ref in trainval} | {
            ref.identity for ref in test
        }
        self.assertEqual(combined, {ref.identity for ref in pool})
        self.assertEqual(len(trainval) + len(test), len(pool))

    def test_ratio_rounding(self) -> None:
        pool = make_pool(100)
        trainval, test = partition_trainval_test(
            pool, seed=1, test_ratio=0.2
        )
        self.assertEqual(len(test), 20)
        self.assertEqual(len(trainval), 80)

    def test_explicit_test_size_is_not_part_of_common_api(self) -> None:
        with self.assertRaises(TypeError):
            partition_trainval_test(make_pool(100), seed=1, test_size=10)  # type: ignore[call-arg]

    def test_default_ratio_is_supported(self) -> None:
        trainval, test = partition_trainval_test(make_pool(10), seed=1)
        self.assertEqual(len(trainval) + len(test), 10)

    def test_rejects_invalid_ratio(self) -> None:
        with self.assertRaises(ValueError):
            partition_trainval_test(make_pool(10), seed=1, test_ratio=0.0)

    def test_rejects_tiny_pool_with_ratio(self) -> None:
        pool = make_pool(1)
        with self.assertRaises(ValueError):
            partition_trainval_test(pool, seed=1, test_ratio=0.2)


class TestRequestedTrainSizesFromRatios(unittest.TestCase):
    """Tests for the authoritative ratio-to-count helper."""

    def test_ratio_to_counts(self) -> None:
        sizes = get_requested_train_sizes_from_ratios(100, [0.2, 0.5, 1.0])
        self.assertEqual(sizes, [20, 50, 100])

    def test_max_train_size_cap_scales_ratios_down(self) -> None:
        # Largest requested size (round(1.0 * 100) = 100) exceeds the cap, so
        # all ratios are scaled proportionally so the largest maps to 40.
        sizes = get_requested_train_sizes_from_ratios(
            100, [0.2, 0.5, 1.0], max_train_size=40
        )
        self.assertEqual(sizes, [8, 20, 40])

    def test_max_train_size_above_pool_is_noop(self) -> None:
        sizes = get_requested_train_sizes_from_ratios(
            100, [0.2, 0.5, 1.0], max_train_size=500
        )
        self.assertEqual(sizes, [20, 50, 100])

    def test_rejects_collapsing_ratios(self) -> None:
        sizes = get_requested_train_sizes_from_ratios(10, [0.10, 0.12])
        self.assertEqual(sizes, [1, 1])


class TestGenerateRandomTrajectory(unittest.TestCase):
    """Tests for :func:`generate_random_trajectory` and end-to-end flow."""

    def test_seed_reproducibility(self) -> None:
        pool = make_pool(20)
        first = generate_random_trajectory(
            seed=3, pool=pool, requested_train_sizes=[2, 4, 8]
        )
        second = generate_random_trajectory(
            seed=3, pool=pool, requested_train_sizes=[2, 4, 8]
        )
        self.assertEqual(
            [ref.identity for ref in first.selected_frames],
            [ref.identity for ref in second.selected_frames],
        )

    def test_selected_frames_is_permutation_prefix(self) -> None:
        pool = make_pool(20)
        trajectory = generate_random_trajectory(
            seed=3, pool=pool, requested_train_sizes=[2, 4, 8]
        )
        self.assertEqual(len(trajectory.selected_frames), 8)
        selected_identities = [
            ref.identity for ref in trajectory.selected_frames
        ]
        self.assertEqual(len(set(selected_identities)), 8)
        pool_identities = {ref.identity for ref in pool}
        self.assertTrue(set(selected_identities) <= pool_identities)

    def test_nested_prefixes_and_validation(self) -> None:
        pool = make_pool(20)
        trajectory = generate_random_trajectory(
            seed=3, pool=pool, requested_train_sizes=[2, 4, 8]
        )
        train_2 = {ref.identity for ref in trajectory.get_train_set(0)}
        train_4 = {ref.identity for ref in trajectory.get_train_set(1)}
        train_8 = {ref.identity for ref in trajectory.get_train_set(2)}
        self.assertTrue(train_2 <= train_4 <= train_8)

        validation = trajectory.get_val_set(1)
        validation_identities = {ref.identity for ref in validation}
        self.assertEqual(len(validation), len(pool) - 4)
        self.assertTrue(validation_identities.isdisjoint(train_4))

    def test_trajectory_has_no_entropy_profile(self) -> None:
        pool = make_pool(20)
        trajectory = generate_random_trajectory(
            seed=3, pool=pool, requested_train_sizes=[2, 4, 8]
        )
        self.assertEqual(trajectory.method, "random")
        self.assertIsNone(trajectory.entropy_profile)

    def test_end_to_end_splitted_data_schema(self) -> None:
        frames = {
            "a.extxyz": [0, 1, 2, 3, 4],
            "b.extxyz": [0, 1, 2, 3, 4],
        }
        pool = get_references_from_frames(frames, "d")
        trainval, test = partition_trainval_test(
            pool, seed=7, test_ratio=0.2
        )
        # total = 10 -> test = 2, trainval = 8.
        self.assertEqual(len(test), 2)
        self.assertEqual(len(trainval), 8)

        sizes = get_requested_train_sizes_from_ratios(
            len(trainval), [0.25, 0.5, 1.0]
        )
        self.assertEqual(sizes, [2, 4, 8])

        trajectory = generate_random_trajectory(
            seed=3,
            pool=trainval,
            requested_train_sizes=sizes,
        )
        self.assertIsInstance(trajectory, TrainValSplitTrajectory)

        schema = SplitDataSchema(
            domain="d",
            grouping_strategy="all",
            group_name="all",
            test_set=test,
            test_ratio=0.2,
            trainval_test_split_seed=7,
            train_val_split_trajectory=trajectory,
        )
        # The persisted schema round-trips and revalidates.
        restored = SplitDataSchema.from_dict(schema.as_dict())
        self.assertEqual(restored, schema)
        self.assertEqual(
            {ref.identity for ref in trajectory.selected_frames}
            | {ref.identity for ref in trajectory.additional_trainval_frames},
            {ref.identity for ref in trainval},
        )
        self.assertEqual(
            [ref.identity for ref in trajectory.additional_trainval_frames],
            [ref.identity for ref in trainval if ref.identity not in {x.identity for x in trajectory.selected_frames}],
        )


if __name__ == "__main__":
    unittest.main()
