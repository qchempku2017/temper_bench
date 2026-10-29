"""Create local bundles from split datasets and catalog models."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter
from ruamel.yaml import YAML

from temper.job_management.paths import bundle_directory, bundle_root
from temper.job_management.storage_utils import atomic_json, store_lock
from temper.mlff.spec_builders import mlff_spec_builder_factory
from temper.mlff.spec_builders.base import pretrained_model_defaults
from temper.schemas.mlff_spec import MLFFSpec
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.schemas.train_unit import TrainingUnit, load_training_units
from temper.utils.defaults import (
    DEFAULT_MLFF_PRETRAINED_MODELS_DIR,
    DEFAULT_SPLIT_RESULTS_DIR,
    DEFAULT_TRAINING_UNITS_FILE,
)
from temper.utils.file_hashing import hashing_session

logger = logging.getLogger(__name__)


class _ModelOverrides(BaseModel):
    """Check the two parameter dictionaries supplied for one model
    variant."""

    # Builders accept parameter dictionaries, but do not read the
    # override file.
    # This small model checks that file's structure: catch misspelled
    # section
    # names, reject non-dictionary sections, and supply an empty
    # dictionary when
    # a section is omitted. It replaces repeated manual checks in the
    # loader.
    # The builders still decide which parameter names and values are
    # valid for
    # each MLFF; this class does not duplicate their model-specific
    # rules.
    model_config = ConfigDict(extra="forbid")
    training_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    testing_parameters: dict[str, JsonValue] = Field(default_factory=dict)


def add_create_bundles_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the command that pairs split datasets with catalog
    models.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Top-level command collection. Registration does not read
        datasets or pretrained weights and does not create directories.
    """
    command = subparsers.add_parser(
        "create_bundles",
        help="Create bundles for selected split datasets and model variants.",
        description=(
            "Combine every selected training unit with every requested model"
            " variant. Write bundle.json in each model/domain/bundle"
            " directory. Zero-shot units are always included for selected"
            " domains. Inputs are read and verified, but are not copied and no"
            " jobs are launched. Use job submit with the same"
            " --bundle-write-root to run the bundles."
        ),
    )
    command.add_argument(
        "split_results",
        nargs="?",
        type=Path,
        default=Path(DEFAULT_SPLIT_RESULTS_DIR),
        help=(
            "SplitConfig.output_path containing domain directories. Defaults"
            " to DEFAULT_SPLIT_RESULTS_DIR (./split_results if unset)."
            " Relative paths use the current working directory."
        ),
    )
    command.add_argument(
        "--model",
        nargs=2,
        action="append",
        required=True,
        metavar=("FAMILY", "VARIANT"),
        help=(
            "Family and variant keys from default_pretrained_models.json, for"
            " example --model mace mpa-0-medium. Repeat this option for"
            " additional models."
        ),
    )
    command.add_argument(
        "--domains",
        nargs="+",
        metavar="NAME",
        help=(
            "Use only these domain directory names. By default, use all"
            " immediate subdirectories containing the configured training-unit"
            " JSON file."
        ),
    )
    command.add_argument(
        "--max-train-n-atoms",
        type=int,
        metavar="N",
        help=(
            "Include fine-tuning units with at most N total atoms across their"
            " training frames. N must be nonnegative; zero selects only"
            " zero-shot units. Combines with --domains. By default, impose no"
            " atom limit."
        ),
    )
    command.add_argument(
        "--pretrained-model-dir",
        type=Path,
        default=Path(DEFAULT_MLFF_PRETRAINED_MODELS_DIR),
        metavar="DIR",
        help=(
            "Directory containing the catalog's pretrained weight files."
            " Defaults to DEFAULT_MLFF_PRETRAINED_MODELS_DIR"
            " (./pretrained_models if unset). Relative paths use the current"
            " working directory. Weights must already exist."
        ),
    )
    command.add_argument(
        "--bundle-write-root",
        type=Path,
        metavar="DIR",
        help=(
            "Create model/domain/bundle directories here. Defaults to"
            " DEFAULT_BUNDLE_WRITE_ROOT (./bundles if unset). Relative paths"
            " use the current working directory. Existing bundle IDs are"
            " skipped."
        ),
    )
    command.add_argument(
        "--model_override_config",
        type=Path,
        metavar="FILE",
        help=(
            "Optional JSON or YAML file keyed by MLFF family, then variant,"
            " containing overriding training_parameters and testing_parameters"
            " dictionaries for the existing spec builders. Training overrides"
            " apply only to fine-tuning units. Omitted settings use spec"
            " builder defaults. Relative file paths use the current working"
            " directory."
        ),
    )


def _load_overrides(
    path: Path | None,
    models: list[tuple[str, str]],
) -> dict[str, dict[str, _ModelOverrides]]:
    """Check configuration structure; leave model parameter validation
    to builders."""
    if path is None:
        return {}
    data = YAML(typ="safe").load(path.expanduser().read_text(encoding="utf-8"))
    # Validate family -> variant -> parameter sections together so
    # errors
    # identify the location of an invalid value within the
    # configuration.
    overrides = TypeAdapter(
        dict[str, dict[str, _ModelOverrides]]
    ).validate_python(data)
    for family, variants in overrides.items():
        for variant in variants:
            pretrained_model_defaults(family, variant)
            if (family, variant) not in models:
                raise ValueError(
                    f"Overrides specify unselected model {family} {variant}."
                )
    return overrides


@hashing_session()
def create_bundles_cli(
    split_results: Path,
    model_keys: list[tuple[str, str]],
    *,
    domains: list[str] | None = None,
    max_train_n_atoms: int | None = None,
    pretrained_model_dir: Path = Path(DEFAULT_MLFF_PRETRAINED_MODELS_DIR),
    bundle_write_root: Path | None = None,
    model_override_config: Path | None = None,
) -> list[Path]:
    """Write one local bundle for each selected training-unit/model
    pair.

    Parameters
    ----------
    split_results : Path
        Output root from data splitting. Read
        DEFAULT_TRAINING_UNITS_FILE in each selected domain directory.
        Dataset paths are resolved beneath this root, replacing the
        saved units' previous root_path values. The resulting bundles
        store an absolute dataset root.
    model_keys : list[tuple[str, str]]
        Nonempty list of MLFF family and variant keys. Every selected
        unit is paired with every model; repeated bundle IDs are
        included only once.
    domains : list[str] or None, optional
        Exact domain directory names. None selects all immediate
        directories containing a training-unit file. Explicit names must
        exist.
    max_train_n_atoms : int or None, optional
        Inclusive limit on total training atoms across frames, not atoms
        per frame. None imposes no limit. Must be nonnegative. Zero-shot
        units in selected domains are always included, including when
        the limit is zero.
    pretrained_model_dir : Path, optional
        Root containing MLFF family checkpoint files; defaults to
        DEFAULT_MLFF_PRETRAINED_MODELS_DIR. Weights are read and hashed,
        not downloaded. Relative paths use the current working
        directory.
    bundle_write_root : Path or None, optional
        Output root for bundle_directory's model/domain/bundle layout.
        None uses DEFAULT_BUNDLE_WRITE_ROOT. Relative paths use the
        current working directory. Each bundle is named bundle.json and
        has files=None; job submit later copies inputs into separate
        attempt directories.
    model_override_config : Path or None, optional
        JSON or YAML mapping from MLFF family to variant to
        training_parameters and testing_parameters dictionaries. None
        uses spec builder defaults. Training overrides apply only to
        fine-tuning units; zero-shot specs always have
        training_parameters=None. The builders merge and validate the
        options using their existing rules. Entries must select
        requested models. Relative file paths use the current working
        directory.

    Returns
    -------
    list[Path]
        Absolute bundle paths in deterministic unit/model order,
        including existing bundles skipped because their bundle ID
        already exists. No input datasets or weights are copied and no
        jobs are submitted.

    Raises
    ------
    ValueError
        If selections, catalog keys, overrides or records are invalid,
        no units match, dataset hashes changed, or a selected unit lacks
        dataset hashes. All selected inputs are checked before writing
        bundles.
    OSError
        If inputs cannot be read or output directories or bundles cannot
        be written. A write failure may leave earlier completed bundles
        in place; repeating the command skips those IDs.
    ruamel.yaml.YAMLError
        If the override file cannot be parsed as JSON or YAML.
    """
    if max_train_n_atoms is not None and max_train_n_atoms < 0:
        raise ValueError("max_train_n_atoms must be nonnegative.")
    model_keys = list(dict.fromkeys(model_keys))
    if not model_keys:
        raise ValueError("Specify at least one model family and variant.")
    for family, variant in model_keys:
        pretrained_model_defaults(family, variant)
    overrides = _load_overrides(model_override_config, model_keys)
    split_results = split_results.expanduser().resolve()
    if not split_results.is_dir():
        raise ValueError(
            f"Split-results directory does not exist: {split_results}"
        )
    selected_domains = (
        sorted(set(domains))
        if domains is not None
        else sorted(
            path.name
            for path in split_results.iterdir()
            if path.is_dir() and (path / DEFAULT_TRAINING_UNITS_FILE).is_file()
        )
    )
    units: dict[str, TrainingUnit] = {}
    for domain in selected_domains:
        if (
            domain in {"", ".", ".."}
            or "/" in domain
            or "\\" in domain
            or Path(domain).is_absolute()
        ):
            raise ValueError(
                f"Expected a domain directory name, got {domain!r}."
            )
        path = split_results / domain / DEFAULT_TRAINING_UNITS_FILE
        for unit in load_training_units(path, verify_files=False):
            if unit.domain != domain:
                raise ValueError(
                    f"Unit domain {unit.domain!r} does not match directory"
                    f" {domain!r}."
                )
            if (
                unit.unit_type == "finetune"
                and max_train_n_atoms is not None
                and unit.train_n_atoms > max_train_n_atoms
            ):
                continue
            if not unit.dataset_sha256:
                raise ValueError(
                    f"Training unit {unit.training_unit_id} has no dataset"
                    " hashes; regenerate the split results."
                )
            data = unit.model_dump()
            data["root_path"] = split_results
            verified = TrainingUnit.model_validate(data)
            units[str(verified.training_unit_id)] = verified
    if not units:
        raise ValueError(
            "No training units match the selected domains and atom limit."
        )
    specs: dict[tuple[str, str, str], MLFFSpec] = {}
    bundles: dict[str, MLFFTrainBundle] = {}
    for unit_id in sorted(units):
        unit = units[unit_id]
        for family, variant in model_keys:
            key = (family, variant, unit.unit_type)
            if key not in specs:
                options = overrides.get(family, {}).get(
                    variant, _ModelOverrides()
                )
                specs[key] = mlff_spec_builder_factory(family)(
                    variant=variant,
                    pretrained_model_dir=(
                        pretrained_model_dir.expanduser().resolve()
                    ),
                    training_parameters=(
                        options.training_parameters
                        if unit.unit_type == "finetune"
                        else None
                    ),
                    testing_parameters=options.testing_parameters,
                ).build()
            # Bundle files not populated until submission,
            # in which bundle writer is called, and it copies inputs and
            # configs into the bundle folder.
            bundle = MLFFTrainBundle(training_unit=unit, mlff_spec=specs[key])
            bundles[str(bundle.mlff_train_bundle_id)] = bundle
    root = bundle_root(bundle_write_root)
    paths: list[Path] = []
    created = 0
    with store_lock(root):
        existing_ids = {
            str(
                MLFFTrainBundle.model_validate_json(
                    path.read_text(encoding="utf-8"),
                    context={"verify_files": False},
                ).mlff_train_bundle_id
            ): path
            for path in sorted(root.glob("*/*/bundle__*/bundle.json"))
        }
        for identity, bundle in bundles.items():
            if identity in existing_ids:
                paths.append(existing_ids[identity])
                continue
            path = bundle_directory(root, bundle) / "bundle.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(path, bundle.model_dump(mode="json"))
            paths.append(path)
            created += 1
    logger.info(
        "Created %d bundles; skipped %d existing IDs. Bundle root: %s",
        created,
        len(paths) - created,
        root,
    )
    return paths
