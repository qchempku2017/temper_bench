"""Tests for the authoritative reconstruction and extxyz export APIs."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from src.temper.schemas import FrameReference, SplitDataSchema
from src.temper.splitting import (
    SourceResolver,
    build_export_filename,
    get_references_from_frames,
    generate_random_trajectory,
    load_frames_from_references,
    load_frames_test,
    load_frames_train_validation,
    partition_trainval_test,
    write_all_sets_in_split_schema_to_extxyz,
    write_single_dataset_to_extxyz,
)


def labeled(index: int, *, stress: bool = False) -> Atoms:
    atoms = Atoms("H2", positions=[[0, 0, index], [0, 0, index + 1]], cell=[10] * 3, pbc=True)
    kwargs = {"energy": -float(index), "forces": np.array([[1, 0, index], [0, 1, -index]], dtype=float)}
    if stress:
        kwargs["stress"] = np.arange(6, dtype=float) + index
    atoms.calc = SinglePointCalculator(atoms, **kwargs)
    atoms.info["tag"] = f"frame{index}"
    return atoms


def schema_for(domain: str = "d1") -> SplitDataSchema:
    files = {"a.extxyz": [0, 1, 2], "b.extxyz": [0, 1]} if domain == "d1" else {"a.extxyz": [0, 1, 2, 3, 4]}
    pool = get_references_from_frames(files, domain)
    trainval, test = partition_trainval_test(pool, seed=7, test_ratio=0.2)
    trajectory = generate_random_trajectory(seed=3, pool=trainval, requested_train_sizes=[1, 2, 4])
    return SplitDataSchema(domain=domain, grouping_strategy="all", group_name="g0", test_set=test, test_ratio=0.2, trainval_test_split_seed=7, train_val_split_trajectory=trajectory)


class Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "data"
        self.out = Path(self.tmp.name) / "out"
        (self.root / "d1").mkdir(parents=True)
        (self.root / "d2").mkdir(parents=True)
        write(self.root / "d1" / "a.extxyz", [labeled(i) for i in range(3)], format="extxyz")
        write(self.root / "d1" / "b.extxyz", [labeled(i) for i in range(3, 5)], format="extxyz")
        write(self.root / "d2" / "a.extxyz", [labeled(i, stress=True) for i in range(5)], format="extxyz")
        unlabeled = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
        write(self.root / "d1" / "unlabeled.extxyz", unlabeled, format="extxyz")

    def tearDown(self) -> None:
        self.tmp.cleanup()


class TestResolverAndLoading(Fixture):
    def test_resolution_cache_and_order(self) -> None:
        resolver = SourceResolver(self.root)
        ref = FrameReference(domain="d1", filename="a.extxyz", frame_index=0)
        self.assertEqual(resolver.resolve_source_path(ref), (self.root / "d1" / "a.extxyz").resolve())
        first = resolver.load_raw_frames(resolver.resolve_source_path(ref))
        second = resolver.load_raw_frames(resolver.resolve_source_path(ref))
        self.assertIs(first, second)
        frames = load_frames_from_references([FrameReference(domain="d1", filename="b.extxyz", frame_index=0), FrameReference(domain="d1", filename="a.extxyz", frame_index=2)], self.root)
        self.assertEqual([f.get_potential_energy() for f in frames], [-3.0, -2.0])

    def test_guard_and_labels(self) -> None:
        with self.assertRaises(NotADirectoryError):
            SourceResolver(self.root / "missing")
        resolver = SourceResolver(self.root)
        with self.assertRaises(ValueError):
            resolver.resolve_source_path(FrameReference(domain="../d2", filename="a.extxyz", frame_index=0))
        frame = load_frames_from_references([FrameReference(domain="d1", filename="a.extxyz", frame_index=1)], self.root)[0]
        self.assertEqual(frame.info["tag"], "frame1")
        np.testing.assert_allclose(frame.get_forces(), [[1, 0, 1], [0, 1, -1]])
        with self.assertRaises(IndexError):
            load_frames_from_references([FrameReference(domain="d1", filename="a.extxyz", frame_index=3)], self.root)
        with self.assertRaises(ValueError):
            load_frames_from_references([FrameReference(domain="d1", filename="unlabeled.extxyz", frame_index=0)], self.root)

    def test_test_and_train_validation_use_checkpoint_index(self) -> None:
        schema = schema_for()
        self.assertEqual(len(load_frames_test(schema, self.root)), 1)
        train, validation = load_frames_train_validation(schema, 1, self.root)
        trajectory = schema.train_val_split_trajectory
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 2)
        expected_refs = trajectory.get_train_set(1)
        self.assertEqual([f.get_potential_energy() for f in train], [-3.0, -2.0])
        self.assertEqual({f.get_potential_energy() for f in train + validation}, {-3.0, -2.0, -1.0, -0.0})
        self.assertEqual(len(expected_refs), len(train))

    def test_stress_and_shared_cache_labels(self) -> None:
        ref = FrameReference(domain="d2", filename="a.extxyz", frame_index=0)
        a, b = load_frames_from_references([ref, ref], self.root)
        np.testing.assert_allclose(a.get_stress(), np.arange(6, dtype=float))
        self.assertIs(a.calc, b.calc)
        a.calc.results["forces"][0, 0] = 99
        self.assertEqual(b.calc.results["forces"][0, 0], 99)


class TestExport(Fixture):
    def test_filename_and_validation(self) -> None:
        self.assertEqual(build_export_filename(domain="d1", group_name="g0", grouping_strategy="all", method="random", role="train", structure_count=2), "d1__all__g0__random__train__n2.extxyz")
        self.assertEqual(build_export_filename(domain="D1/x", group_name="g 0", grouping_strategy=None, method="random", role="train", structure_count=5), "D1_x__unknown_grouping__g_0__random__train__n5.extxyz")
        with self.assertRaises(ValueError):
            build_export_filename(domain="d1", group_name="g0", grouping_strategy=None, method="random", role="bad", structure_count=1)

    def test_single_roundtrip_replaces_generated_artifact(self) -> None:
        schema = schema_for()
        train, _ = load_frames_train_validation(schema, 1, self.root)
        kwargs = dict(domain="d1", group_name="g0", grouping_strategy="all", method="random", role="train", output_dir=self.out)
        path = write_single_dataset_to_extxyz(train, **kwargs)
        self.assertEqual([f.get_potential_energy() for f in read(path, index=":")], [f.get_potential_energy() for f in train])
        self.assertEqual(write_single_dataset_to_extxyz(train, **kwargs), path)
        with self.assertRaises(ValueError):
            write_single_dataset_to_extxyz([], **kwargs)

    def test_all_sets_exports_expected_files_and_collision_policy(self) -> None:
        schema = schema_for()
        written = write_all_sets_in_split_schema_to_extxyz(schema, self.out, self.root)
        self.assertEqual(len(written["train"]), 3)
        self.assertEqual(written["validation"], [])
        self.assertEqual(written["test"][0].name, "d1__all__g0__random__test__n1.extxyz")
        self.assertEqual([p.name for p in written["train"]], ["d1__all__g0__random__train__n1.extxyz", "d1__all__g0__random__train__n2.extxyz", "d1__all__g0__random__train__n4.extxyz"])
        for path in written["train"] + written["test"]:
            self.assertTrue(path.is_file())
            self.assertGreater(len(read(path, index=":")), 0)
        self.assertEqual(write_all_sets_in_split_schema_to_extxyz(schema, self.out, self.root), written)

    def test_stress_roundtrip(self) -> None:
        schema = schema_for("d2")
        written = write_all_sets_in_split_schema_to_extxyz(schema, self.out, self.root)
        for frame in read(written["train"][0], index=":") + read(written["test"][0], index=":"):
            self.assertEqual(len(frame.get_stress()), 6)


    def test_can_export_validation_sets_when_requested(self) -> None:
        schema = schema_for()
        written = write_all_sets_in_split_schema_to_extxyz(
            schema, self.out, self.root, write_validation=True
        )
        self.assertEqual(len(written["validation"]), 2)
        for path in written["validation"]:
            self.assertTrue(path.is_file())
            self.assertGreater(len(read(path, index=":")), 0)


if __name__ == "__main__":
    unittest.main()
