"""Defines default file-system locations and splitting parameters used across TemPER. Values are read from environment variables at import time when provided."""
from __future__ import annotations

import os


def _env_float(name: str, default: float) -> float:
    """Read ``name`` as a float, falling back to ``default``.

    An unset, empty, or whitespace-only variable falls back to ``default``.
    A non-empty value is parsed with :func:`float`; an unparseable value
    raises a ``ValueError`` naming the variable.

    Parameters
    ----------
    name : str
        Name of the environment variable.
    default : float
        Fallback value when the variable is unset or empty.

    Returns
    -------
    float
        The parsed value.

    Raises
    ------
    ValueError
        If the variable is set to a non-empty value that cannot be parsed as
        a float.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name!r} must be a float, got {raw!r}."
        ) from exc


def _env_int(name: str, default: int) -> int:
    """Read ``name`` as an int, falling back to ``default``.

    An unset, empty, or whitespace-only variable falls back to ``default``.
    A non-empty value is parsed with :func:`int`; an unparseable value raises
    a ``ValueError`` naming the variable.

    Parameters
    ----------
    name : str
        Name of the environment variable.
    default : int
        Fallback value when the variable is unset or empty.

    Returns
    -------
    int
        The parsed value.

    Raises
    ------
    ValueError
        If the variable is set to a non-empty value that cannot be parsed as
        an int.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name!r} must be an int, got {raw!r}."
        ) from exc


##################################
# Default values for data storage.
##################################
DEFAULT_DATA_DIR: str = os.environ.get("DEFAULT_DATA_DIR", "./data")
# Default storage directory for training units.
DEFAULT_SPLIT_RESULTS_DIR: str = os.environ.get("DEFAULT_SPLIT_RESULTS_DIR", "./split_results")
# Default name of the info file under each data directory.
DEFAULT_METADATA_FILE: str = os.environ.get("DEFAULT_METADATA_FILE", "metadata.json")
# Default name of the output file to save grouped domains in a domain.
DEFAULT_GROUPED_DOMAIN_FILE: str = os.environ.get("DEFAULT_GROUPED_DOMAIN_FILE", "grouped_domains.json")
# Default name of the output file to save split groups in a domain.
DEFAULT_SPLIT_GROUPS_FILE: str = os.environ.get("DEFAULT_SPLIT_GROUPS_FILE", "split_groups.json")
# Default name of the output file to save training units in a domain.
DEFAULT_TRAINING_UNITS_FILE: str = os.environ.get("DEFAULT_TRAINING_UNITS_FILE", "training_units.json")

##################################
# Default values for creating experiments from extxyz data.
##################################
# Default ratio of test set size to total dataset size (train + val + test).
DEFAULT_TEST_RATIO: float = _env_float("DEFAULT_TEST_RATIO", 0.2)
# Default ratios of training set size to train + val dataset size.
_DEFAULT_TRAIN_RATIOS: list[float] = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
_raw_train_ratios = os.environ.get("DEFAULT_TRAIN_RATIOS")
if _raw_train_ratios is None or not _raw_train_ratios.strip():
    DEFAULT_TRAIN_RATIOS: list[float] = _DEFAULT_TRAIN_RATIOS
else:
    DEFAULT_TRAIN_RATIOS = [
        float(ratio.strip())
        for ratio in _raw_train_ratios.split(",")
        if ratio.strip()
    ]
# Default maximum number of training data points. If maximum training set exceeds this,
#  the training set ratios will be scaled down proportionally.
DEFAULT_MAX_N_TRAIN: int = _env_int("DEFAULT_MAX_N_TRAIN", 3000)
# Default maximum number of test data points. If maximum test set exceeds this,
#  the test set ratio will be scaled down proportionally.
#  Cross test sets will also not exceed this size.
DEFAULT_SPLIT_REPEATS: int = _env_int("DEFAULT_SPLIT_REPEATS", 3)


__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_SPLIT_RESULTS_DIR",
    "DEFAULT_METADATA_FILE",
    "DEFAULT_MAX_N_TRAIN",
    "DEFAULT_METADATA_FILE",
    "DEFAULT_TEST_RATIO",
    "DEFAULT_TRAIN_RATIOS",
    "DEFAULT_SPLIT_REPEATS",
    "DEFAULT_GROUPED_DOMAIN_FILE",
    "DEFAULT_SPLIT_GROUPS_FILE",
    "DEFAULT_TRAINING_UNITS_FILE",
    "_env_float",
    "_env_int",
]
