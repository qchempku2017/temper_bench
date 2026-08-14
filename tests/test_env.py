"""Tests for environment-variable type conversion in ``src.temper.utils.env``.

The module reads numeric environment variables once at import time and must
convert them to the documented Python types (``float`` for ratios, ``int`` for
counts) rather than leaking raw strings, and must reject unparseable values.
Each test that reloads the module restores the clean-environment values so the
module is left in its default state for the rest of the suite.
"""
from __future__ import annotations

import importlib
import os
import unittest
from unittest import mock

import src.temper.utils.env as env_module


class TestEnvTypeConversion(unittest.TestCase):
    """Tests for :mod:`src.temper.utils.env` type conversion."""

    def test_defaults_are_typed(self) -> None:
        self.assertIsInstance(env_module.DEFAULT_TEST_RATIO, float)
        self.assertIsInstance(env_module.DEFAULT_MAX_N_TRAIN, int)
        self.assertIsInstance(env_module.DEFAULT_MAX_N_TEST, int)
        self.assertTrue(
            all(isinstance(ratio, float) for ratio in env_module.DEFAULT_TRAIN_RATIOS)
        )
        self.assertEqual(env_module.DEFAULT_TEST_RATIO, 0.2)
        self.assertEqual(env_module.DEFAULT_MAX_N_TRAIN, 3000)
        self.assertEqual(env_module.DEFAULT_MAX_N_TEST, 1000)

    def test_custom_values_are_converted(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "DEFAULT_TEST_RATIO": "0.5",
                "DEFAULT_TRAIN_RATIOS": "0.25, 0.75",
                "DEFAULT_MAX_N_TRAIN": "123",
                "DEFAULT_MAX_N_TEST": "456",
            },
        ):
            reloaded = importlib.reload(env_module)
            self.assertEqual(reloaded.DEFAULT_TEST_RATIO, 0.5)
            self.assertEqual(reloaded.DEFAULT_TRAIN_RATIOS, [0.25, 0.75])
            self.assertEqual(reloaded.DEFAULT_MAX_N_TRAIN, 123)
            self.assertEqual(reloaded.DEFAULT_MAX_N_TEST, 456)
        importlib.reload(env_module)  # restore clean-environment defaults.

    def test_empty_values_fall_back_to_defaults(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "DEFAULT_TEST_RATIO": "",
                "DEFAULT_TRAIN_RATIOS": "  ",
                "DEFAULT_MAX_N_TRAIN": "   ",
            },
        ):
            reloaded = importlib.reload(env_module)
            self.assertEqual(reloaded.DEFAULT_TEST_RATIO, 0.2)
            self.assertEqual(reloaded.DEFAULT_TRAIN_RATIOS, [0.1, 0.2, 0.4, 0.6, 0.8, 0.95])
            self.assertEqual(reloaded.DEFAULT_MAX_N_TRAIN, 3000)
        importlib.reload(env_module)

    def test_invalid_float_raises(self) -> None:
        with mock.patch.dict(os.environ, {"DEFAULT_TEST_RATIO": "abc"}):
            with self.assertRaises(ValueError):
                importlib.reload(env_module)
        importlib.reload(env_module)

    def test_invalid_int_raises(self) -> None:
        with mock.patch.dict(os.environ, {"DEFAULT_MAX_N_TRAIN": "12.5"}):
            with self.assertRaises(ValueError):
                importlib.reload(env_module)
        importlib.reload(env_module)

    def test_invalid_train_ratio_raises(self) -> None:
        with mock.patch.dict(os.environ, {"DEFAULT_TRAIN_RATIOS": "0.1, nope"}):
            with self.assertRaises(ValueError):
                importlib.reload(env_module)
        importlib.reload(env_module)

    def test_helper_functions_reject_unparseable_values(self) -> None:
        # Direct helper contract: unset falls back, unparseable raises.
        self.assertEqual(env_module._env_int("DEFINITELY_UNSET_VAR", 7), 7)
        with mock.patch.dict(os.environ, {"UNPARSEABLE_INT": "3.7"}):
            with self.assertRaises(ValueError):
                env_module._env_int("UNPARSEABLE_INT", 7)
        with mock.patch.dict(os.environ, {"UNPARSEABLE_FLOAT": "x"}):
            with self.assertRaises(ValueError):
                env_module._env_float("UNPARSEABLE_FLOAT", 0.2)


if __name__ == "__main__":
    unittest.main()
