"""Readable, portable bundle directory names."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from temper.schemas.mlff_train_bundle import MLFFTrainBundle


# Comment: this had better be a default envvar in utils.defaults.
#   Just import from there.
def bundle_root(value: str | Path | None = None) -> Path:
    """Resolve an explicit or environment-configured bundle root.

    Parameters
    ----------
    value : str, Path or None, optional
        Explicit root, overriding DEFAULT_BUNDLE_WRITE_ROOT. The fallback is
        ./bundles. Relative paths use the current working directory.

    Returns
    -------
    Path
        Absolute path; no directory is created.
    """
    root = Path(value or os.environ.get("DEFAULT_BUNDLE_WRITE_ROOT", "./bundles")).expanduser().resolve()
    # Explicit Windows long-path syntax works without changing machine policy.
    # A readable nested experiment path can otherwise exceed MAX_PATH once
    # native model output names are appended.
    if os.name == "nt" and not str(root).startswith("\\\\?\\"):
        name = str(root)
        root = Path("\\\\?\\UNC\\" + name[2:] if name.startswith("\\\\") else "\\\\?\\" + name)
    return root


def _component(value: object, limit: int = 32) -> str:
    """Make a safe, readable directory name from a value across platforms."""
    original = str(value)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", original).strip(". ") or "unnamed"
    if safe.upper().split(".")[0] in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        safe = "_" + safe
    if safe != original or len(safe) > limit:
        safe = safe[:limit - 9] + "_" + hashlib.sha256(original.encode()).hexdigest()[:8]
    return safe


def bundle_directory(root: Path, bundle: MLFFTrainBundle) -> Path:
    """Build a readable directory containing the full scientific identity.

    Parameters
    ----------
    root : Path
        Configured local store root.
    bundle : MLFFTrainBundle
        Recipe with verified identities.

    Returns
    -------
    Path
        Model/version, domain and bundle directory. The full UUID avoids
        collisions between abbreviated identities. No directory is created.
    """
    unit = bundle.training_unit
    model = bundle.mlff_spec.pretrained_model
    label = "__".join(_component(x, 20) for x in (unit.grouping_strategy, unit.group_name, unit.method))
    return root / f"{_component(model.name)}__{_component(model.version)}" / _component(unit.domain) / (
        f"bundle__{label}__{unit.unit_type}__n{unit.train_n_frames}"
        f"__repeat{unit.repeat_id}__{bundle.mlff_train_bundle_id}"
    )


def next_attempt(directory: Path) -> Path:
    """Choose the next unused attempt name while the caller holds the store lock.

    Parameters
    ----------
    directory : Path
        Directory belonging to one full bundle identity.

    Returns
    -------
    Path
        Uncreated attempt_NNN path. Existing attempts are never overwritten.
    """
    numbers = [int(p.name[8:]) for p in directory.glob("attempt_*") if p.name[8:].isdigit()]
    return directory / f"attempt_{max(numbers, default=0) + 1:03d}"
