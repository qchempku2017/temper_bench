"""Tests for the authoritative persisted split contract."""
from __future__ import annotations

import unittest
from pydantic import ValidationError

from src.temper.schemas import (
    EntropyProfile, EntropyProfilePoint, FrameReference, QuestsSplitConfig,
    SplitDataSchema, SplitSchema, TrainValSplitTrajectory,
)


def ref(i: int, domain: str = "d", filename: str = "a.extxyz") -> FrameReference:
    return FrameReference(domain=domain, filename=filename, frame_index=i)


def trajectory(selected=None, additional=None, sizes=None, profile=None, method="random", seed=1):
    return TrainValSplitTrajectory(
        method=method, seed=seed if method == "random" else None,
        requested_train_sizes=sizes or [2, 4],
        selected_frames=selected or [ref(i) for i in range(4)],
        additional_trainval_frames=additional or [], entropy_profile=profile,
    )


class TestFrameReference(unittest.TestCase):
    def test_valid_and_identity(self):
        value = FrameReference(domain="d", filename="sub/a.extxyz", frame_index=3)
        self.assertEqual(value.identity, ("d", "sub/a.extxyz", 3))

    def test_filename_and_domain_validation(self):
        for filename in ("/a.extxyz", "C:\\a.extxyz", "../a.extxyz", "a.xyz", ""):
            with self.subTest(filename=filename), self.assertRaises(ValidationError):
                ref(0, filename=filename)
        for domain in ("", "   "):
            with self.subTest(domain=domain), self.assertRaises(ValidationError):
                ref(0, domain=domain)
        with self.assertRaises(ValidationError):
            ref(-1)


class TestEntropyProfile(unittest.TestCase):
    def point(self, size):
        return EntropyProfilePoint(training_size=size, cumulative_entropy=.5, information_gain=.1)

    def test_order_and_finite_validation(self):
        profile = EntropyProfile(points=[self.point(2), self.point(4)])
        self.assertEqual([p.training_size for p in profile.points], [2, 4])
        for points in ([], [self.point(2), self.point(2)]):
            with self.subTest(points=points), self.assertRaises(ValidationError):
                EntropyProfile(points=points)

    def test_nonfinite_values_rejected(self):
        with self.assertRaises(ValidationError):
            EntropyProfile(points=[EntropyProfilePoint(training_size=1, cumulative_entropy=float("nan"), information_gain=0)])


class TestTrainValSplitTrajectory(unittest.TestCase):
    def test_profile_must_exactly_match_requested_sizes_and_empty_is_not_a_profile(self):
        profile = EntropyProfile(points=[EntropyProfilePoint(training_size=2, cumulative_entropy=0, information_gain=0)])
        with self.assertRaises(ValidationError):
            trajectory(profile=profile)
        with self.assertRaises(ValidationError):
            EntropyProfile(points=[])

    def test_duplicate_and_overlap_rejected(self):
        with self.assertRaises(ValidationError):
            trajectory(selected=[ref(0), ref(0)], sizes=[1, 2])
        with self.assertRaises(ValidationError):
            trajectory(additional=[ref(0)])
        with self.assertRaises(ValidationError):
            trajectory(additional=[ref(4), ref(4)], sizes=[1, 2])

    def test_accessors_use_natural_requested_size_indexing(self):
        value = trajectory(selected=[ref(i) for i in range(5)], additional=[ref(5), ref(6)], sizes=[1, 3, 5])
        self.assertEqual([x.frame_index for x in value.get_train_set(1)], [0, 1, 2])
        self.assertEqual([x.frame_index for x in value.get_val_set(0)], [1, 2, 3, 4, 5, 6])
        self.assertEqual([x.frame_index for x in value.get_train_set(-1)], [0, 1, 2, 3, 4])
        self.assertEqual([x.frame_index for x in value.get_val_set(-1)], [5, 6])
        self.assertEqual([x.frame_index for x in value.get_train_set(True)], [0, 1, 2])
        for index in (3, 9):
            with self.subTest(index=index), self.assertRaises(IndexError):
                value.get_train_set(index)
        with self.assertRaises(TypeError):
            value.get_val_set("0")

    def test_round_trip_has_singular_serialized_shape(self):
        value = trajectory(additional=[ref(4), ref(5)], sizes=[2, 4])
        dumped = value.model_dump()
        self.assertIn("additional_trainval_frames", dumped)
        self.assertNotIn("trainval_pool", dumped)
        self.assertEqual(TrainValSplitTrajectory.model_validate(dumped), value)


class TestSplitDataSchema(unittest.TestCase):
    def make_schema(self, trajectory_value=None, test=None, ratio=.2):
        test = test or [ref(8), ref(9)]
        selected = [ref(i) for i in range(6)]
        value = trajectory_value or trajectory(selected=selected, additional=[ref(6), ref(7)], sizes=[2, 4, 6])
        return SplitDataSchema(domain="d", grouping_strategy="all", group_name="g", test_set=test,
                               test_ratio=ratio, trainval_test_split_seed=7,
                               train_val_split_trajectory=value)

    def test_singular_contract_and_complete_inventory(self):
        schema = self.make_schema()
        self.assertIsInstance(schema.train_val_split_trajectory, TrainValSplitTrajectory)
        self.assertEqual(len(schema.train_val_split_trajectory.get_val_set(0)), 6)
        dumped = schema.model_dump()
        self.assertNotIn("schema_version", dumped)
        self.assertNotIn("trainval_pool", dumped)
        self.assertNotIsInstance(dumped["train_val_split_trajectory"], list)

    def test_rejects_ratio_domain_overlap_and_outside_trajectory(self):
        with self.assertRaises(ValidationError): self.make_schema(ratio=0)
        with self.assertRaises(ValidationError): self.make_schema(test=[ref(0), ref(9)])
        with self.assertRaises(ValidationError): self.make_schema(test=[ref(8, domain="x"), ref(9)])
        outside = trajectory(selected=[ref(99)], sizes=[1])
        with self.assertRaises(ValidationError): self.make_schema(trajectory_value=outside, test=[ref(8), ref(9)])

    def test_config_and_round_trip(self):
        schema = self.make_schema()
        schema.quests_config = QuestsSplitConfig(device="cpu")
        self.assertEqual(SplitDataSchema.from_dict(schema.as_dict()), schema)


class TestLegacySchema(unittest.TestCase):
    def test_legacy_schema_remains_functional(self):
        value = SplitSchema(domain="d", grouping_strategy="all", group_name="g", train_val_split_method="random",
                            train_val_test_split_seed=1, train_val_split_seed=2,
                            train_set={"a.extxyz": [0]}, val_set={}, test_set={})
        self.assertEqual(value.train_size, 1)


if __name__ == "__main__":
    unittest.main()
