"""Load and type-convert system environment variables.

Environment variables are read once at module import time. String-valued
variables (directories, filenames) are used verbatim; numeric variables are
converted to the documented Python type (``float`` for ratios, ``int`` for
counts) and reject unparseable values with a clear error.

List-valued variables are set as a string of comma-separated values, for
example::

    DEFAULT_TRAIN_RATIOS = "0.1, 0.2, 0.3, 0.4, 0.5"
"""
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
# Default experiment storage directory
DEFAULT_EXP_DIR: str = os.environ.get("DEFAULT_EXP_DIR", "./experiments")
# Default name of the info file under each data directory.
DEFAULT_METADATA_FILE: str = os.environ.get("DEFAULT_METADATA_FILE", "metadata.json")

##################################
# Default values for creating experiments from extxyz data.
##################################
# Default ratio of test set size to total dataset size (train + val + test).
DEFAULT_TEST_RATIO: float = _env_float("DEFAULT_TEST_RATIO", 0.2)
# Default ratios of training set size to train + val dataset size.
_DEFAULT_TRAIN_RATIOS: list[float] = [0.1, 0.2, 0.4, 0.6, 0.8, 0.95]
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
# TODO: this is not yet effective. Should implement this.
DEFAULT_MAX_N_TEST: int = _env_int("DEFAULT_MAX_N_TEST", 1000)


__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_EXP_DIR",
    "DEFAULT_MAX_N_TEST",
    "DEFAULT_MAX_N_TRAIN",
    "DEFAULT_METADATA_FILE",
    "DEFAULT_TEST_RATIO",
    "DEFAULT_TRAIN_RATIOS",
    "_env_float",
    "_env_int",
]
