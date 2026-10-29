"""Check bundle creation and directory-based submission without launching
jobs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from temper.entrypoints.create_bundles import create_bundles_cli
from temper.entrypoints.main import main_parser
from temper.job_management.paths import bundle_directory, bundle_root
from temper.mlff.spec_builders.base import pretrained_model_defaults
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.schemas.train_unit import TrainingUnit
from temper.utils.defaults import DEFAULT_TRAINING_UNITS_FILE


def _write_units(root: Path, units: list[TrainingUnit]) -> None:
    for domain in {unit.domain for unit in units}:
        path = root / domain / DEFAULT_TRAINING_UNITS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                [
                    unit.with_dataset_hashes().model_dump(mode="json")
                    for unit in units
                    if unit.domain == domain
                ]
            ),
            encoding="utf-8",
        )


def _write_weights(root: Path, models: list[tuple[str, str]]) -> None:
    for family, variant in models:
        path = (
            root / pretrained_model_defaults(family, variant)["model_filename"]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test checkpoint")
        if family in {"dpa4", "dpa4c"}:
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "model": {
                            "type_map": ["H", "He"],
                            "descriptor": {"type": family},
                        },
                    }
                )
            )


def test_create_pairs_and_reuses_ids(
    finetune_training_unit: TrainingUnit,
    zeroshot_training_unit: TrainingUnit,
    tmp_path: Path,
) -> None:
    """Create every pairing and retain existing bundles when IDs repeat.

    Parameters
    ----------
    finetune_training_unit, zeroshot_training_unit : TrainingUnit
        Units sharing accessible test datasets in one domain.
    tmp_path : Path
        Temporary root for model weights and bundles.
    """
    unit = finetune_training_unit
    _write_units(unit.root_path, [unit, zeroshot_training_unit, unit])
    models = [("mace", "mpa-0-medium"), ("dpa4", "nano-omat24-v20260805")]
    weights, output = tmp_path / "weights", tmp_path / "bundles"
    _write_weights(weights, models)
    paths = create_bundles_cli(
        unit.root_path,
        models + models,
        pretrained_model_dir=weights,
        bundle_write_root=output,
    )
    assert len(paths) == 4
    for path in paths:
        bundle = MLFFTrainBundle.model_validate_json(path.read_text())
        assert (
            path
            == bundle_directory(bundle_root(output), bundle) / "bundle.json"
        )
        assert bundle.files is None
        assert (bundle.mlff_spec.training_parameters is None) == (
            bundle.unit_type == "zeroshot"
        )
        assert not list(path.parent.glob("attempt_*"))
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths
    }
    assert (
        create_bundles_cli(
            unit.root_path,
            models,
            pretrained_model_dir=weights,
            bundle_write_root=output,
        )
        == paths
    )
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths
    }

    # Submission fills the shared bundle's file locations. Repeating
    # creation
    # must keep these locations so existing attempts can still collect
    # results.
    shared = paths[0]
    bundle = MLFFTrainBundle.model_validate_json(
        shared.read_text(encoding="utf-8")
    )
    _, packaged = bundle.write_submit_folder(shared.parent / "attempt_001")
    shared.write_text(packaged.model_dump_json(), encoding="utf-8")
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths
    }
    assert (
        create_bundles_cli(
            unit.root_path,
            models,
            pretrained_model_dir=weights,
            bundle_write_root=output,
        )
        == paths
    )
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths
    }
    assert (
        MLFFTrainBundle.model_validate_json(
            shared.read_text(encoding="utf-8")
        ).files
        == packaged.files
    )


@pytest.mark.parametrize(
    "limit, expected", [(None, 2), (2, 2), (1, 1), (0, 1)]
)
def test_domain_and_atom_filters_keep_zeroshot(
    finetune_training_unit: TrainingUnit,
    zeroshot_training_unit: TrainingUnit,
    tmp_path: Path,
    limit: int | None,
    expected: int,
) -> None:
    """Combine domain and inclusive atom filters without removing zero-shot
    units.

    Parameters
    ----------
    finetune_training_unit, zeroshot_training_unit : TrainingUnit
        Units with respectively two and zero training atoms.
    tmp_path : Path
        Temporary root for weights and output bundles.
    limit : int or None
        Maximum total training atoms; None means no limit.
    expected : int
        Number of selected units in the requested domain.
    """
    unit = finetune_training_unit
    _write_units(unit.root_path, [unit, zeroshot_training_unit])
    other = unit.root_path / "other"
    other.mkdir()
    (other / DEFAULT_TRAINING_UNITS_FILE).write_text("invalid JSON")
    models = [("mace", "mpa-0-medium")]
    _write_weights(tmp_path / "weights", models)
    paths = create_bundles_cli(
        unit.root_path,
        models,
        domains=[unit.domain],
        max_train_n_atoms=limit,
        pretrained_model_dir=tmp_path / "weights",
        bundle_write_root=tmp_path / "bundles",
    )
    bundles = [
        MLFFTrainBundle.model_validate_json(path.read_text()) for path in paths
    ]
    assert len(bundles) == expected
    assert any(bundle.unit_type == "zeroshot" for bundle in bundles)
    assert all(
        bundle.training_unit.domain == unit.domain for bundle in bundles
    )


def test_overrides_and_relocated_split_root(
    finetune_training_unit: TrainingUnit,
    zeroshot_training_unit: TrainingUnit,
    tmp_path: Path,
) -> None:
    """Use the supplied split root and apply builder overrides to the
    proper modes.

    Parameters
    ----------
    finetune_training_unit, zeroshot_training_unit : TrainingUnit
        Units whose saved dataset root will be removed after relocation.
    tmp_path : Path
        Temporary root for relocated datasets, overrides, weights, and
        bundles.
    """
    unit = finetune_training_unit
    _write_units(unit.root_path, [unit, zeroshot_training_unit])
    relocated = tmp_path / "relocated"
    shutil.copytree(unit.root_path, relocated)
    shutil.rmtree(unit.root_path)
    config = tmp_path / "overrides.yaml"
    config.write_text(
        "mace:\n  mpa-0-medium:\n    training_parameters:\n"
        "      max_num_epochs: 7\n    testing_parameters:\n"
        "      default_dtype: float64\n",
        encoding="utf-8",
    )
    models = [("mace", "mpa-0-medium")]
    _write_weights(tmp_path / "weights", models)
    paths = create_bundles_cli(
        relocated,
        models,
        pretrained_model_dir=tmp_path / "weights",
        bundle_write_root=tmp_path / "bundles",
        model_override_config=config,
    )
    for path in paths:
        bundle = MLFFTrainBundle.model_validate_json(path.read_text())
        assert bundle.training_unit.root_path == relocated.resolve()
        assert (
            bundle.mlff_spec.testing_parameters["default_dtype"] == "float64"
        )
        if bundle.unit_type == "finetune":
            assert bundle.mlff_spec.training_parameters is not None
            assert bundle.mlff_spec.training_parameters["max_num_epochs"] == 7
            assert bundle.mlff_spec.training_parameters["patience"] == 8
        else:
            assert bundle.mlff_spec.training_parameters is None


def test_defaults_and_submit_parser() -> None:
    """Use configured directory defaults and reject the old bundle-list
    syntax."""
    from temper.utils.defaults import (
        DEFAULT_MLFF_PRETRAINED_MODELS_DIR,
        DEFAULT_SPLIT_RESULTS_DIR,
    )

    parser = main_parser()
    args = parser.parse_args(
        ["create_bundles", "--model", "mace", "mpa-0-medium"]
    )
    assert args.split_results == Path(DEFAULT_SPLIT_RESULTS_DIR)
    assert args.pretrained_model_dir == Path(
        DEFAULT_MLFF_PRETRAINED_MODELS_DIR
    )
    assert args.bundle_write_root is None
    args = parser.parse_args(
        ["job", "submit", "--resource-file", "cluster.yaml"]
    )
    assert not hasattr(args, "bundles")
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["job", "submit", "bundle.json", "--resource-file", "cluster.yaml"]
        )
    assert "create_bundles" in parser.format_help()


def test_cli_discovers_all_domains_and_uses_default_output(
    finetune_training_unit: TrainingUnit,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the CLI across two domains and use the configured output root.

    Parameters
    ----------
    finetune_training_unit : TrainingUnit
        Source unit whose datasets are copied into a second domain.
    tmp_path : Path
        Temporary location for the model weights and configured bundle
        root.
    monkeypatch : pytest.MonkeyPatch
        Fixture setting the default output root without altering the
        environment.
    """
    from temper.entrypoints.main import main
    import temper.job_management.paths as paths_module

    unit = finetune_training_unit
    shutil.copytree(
        unit.root_path / unit.domain, unit.root_path / "second_domain"
    )
    second = unit.model_copy(update={"domain": "second_domain"})
    _write_units(unit.root_path, [unit, second])
    weights = tmp_path / "weights"
    _write_weights(weights, [("mace", "mpa-0-medium")])
    output = tmp_path / "configured_bundles"
    monkeypatch.setitem(
        paths_module.__dict__, "DEFAULT_BUNDLE_WRITE_ROOT", str(output)
    )
    with pytest.raises(SystemExit) as result:
        main(
            [
                "create_bundles",
                str(unit.root_path),
                "--model",
                "mace",
                "mpa-0-medium",
                "--pretrained-model-dir",
                str(weights),
            ]
        )
    assert result.value.code == 0
    bundles = list(bundle_root(output).glob("*/*/bundle__*/bundle.json"))
    assert len(bundles) == 2
    assert {
        json.loads(path.read_text())["training_unit"]["domain"]
        for path in bundles
    } == {
        unit.domain,
        "second_domain",
    }


@pytest.mark.parametrize(
    "problem",
    [
        "negative_limit",
        "unknown_model",
        "empty",
        "changed_data",
        "bad_override",
        "null_parameters",
        "list_parameters",
        "list_variants",
    ],
)
def test_invalid_creation_does_not_write_bundles(
    finetune_training_unit: TrainingUnit,
    tmp_path: Path,
    problem: str,
) -> None:
    """Reject invalid inputs before creating output bundles.

    Parameters
    ----------
    finetune_training_unit : TrainingUnit
        Unit with readable datasets to modify for each failure
        condition.
    tmp_path : Path
        Temporary root for weights, overrides, and output.
    problem : str
        Invalid selection, changed dataset, or unsupported override to
        exercise.
    """
    unit = finetune_training_unit
    _write_units(unit.root_path, [unit])
    models = [("mace", "mpa-0-medium")]
    _write_weights(tmp_path / "weights", models)
    config = None
    if problem == "unknown_model":
        models = [("mace", "does-not-exist")]
    if problem == "changed_data":
        assert unit.train_set is not None
        unit.dataset_source(unit.train_set).write_bytes(b"changed")
    invalid_overrides = {
        "bad_override": {"mace": {"mpa-0-medium": {"training_parameter": {}}}},
        "null_parameters": {
            "mace": {"mpa-0-medium": {"training_parameters": None}}
        },
        "list_parameters": {
            "mace": {"mpa-0-medium": {"testing_parameters": []}}
        },
        "list_variants": {"mace": []},
    }
    if problem in invalid_overrides:
        config = tmp_path / "bad.json"
        config.write_text(
            json.dumps(invalid_overrides[problem]), encoding="utf-8"
        )
    output = tmp_path / "bundles"
    with pytest.raises(ValueError):
        create_bundles_cli(
            unit.root_path,
            models,
            max_train_n_atoms=(
                -1
                if problem == "negative_limit"
                else (0 if problem == "empty" else None)
            ),
            pretrained_model_dir=tmp_path / "weights",
            bundle_write_root=output,
            model_override_config=config,
        )
    assert not output.exists()
