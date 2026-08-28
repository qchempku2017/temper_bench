"""Tests for the top-level CLI parser and split orchestration."""
from __future__ import annotations

import logging
from pathlib import Path

from monty.serialization import dumpfn, loadfn
import pytest


def test_main_parser_owns_global_logging_and_progress_options() -> None:
    from temper.entrypoints.main import main_parser
    from temper.utils.defaults import DEFAULT_SPLIT_CONFIG_FILE

    default_args = main_parser().parse_args(["split"])
    assert default_args.command == "split"
    assert default_args.config_file == Path(DEFAULT_SPLIT_CONFIG_FILE)
    assert default_args.log_level == "INFO"
    assert default_args.progress == "auto"

    args = main_parser().parse_args([
        "--verbose",
        "--progress",
        "plain",
        "split",
        "--config-file",
        "custom.yaml",
    ])
    assert args.config_file == Path("custom.yaml")
    assert args.log_level == "DEBUG"
    assert args.progress == "plain"
    assert main_parser().parse_args([
        "--log-level",
        "error",
        "split",
    ]).log_level == "ERROR"
    with pytest.raises(SystemExit):
        main_parser().parse_args(["split", "--root-path", "data"])
    with pytest.raises(SystemExit):
        main_parser().parse_args(["--verbose", "--quiet", "split"])


def test_main_configures_logging_before_subcommand_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import temper.entrypoints.main as entrypoint

    calls: list[tuple] = []
    monkeypatch.setattr(
        entrypoint,
        "configure_cli_logging",
        lambda level, progress_mode: calls.append(
            ("configure", level, progress_mode)
        ),
    )
    monkeypatch.setattr(
        entrypoint,
        "split_cli",
        lambda config_file: calls.append(("split", config_file)) or 0,
    )
    monkeypatch.setattr(
        entrypoint,
        "shutdown_progress",
        lambda: calls.append(("shutdown",)),
    )

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main([
            "--verbose",
            "--progress",
            "plain",
            "split",
            "--config-file",
            "custom.yaml",
        ])

    assert exc_info.value.code == 0
    assert calls == [
        ("configure", "DEBUG", "plain"),
        ("split", Path("custom.yaml")),
        ("shutdown",),
    ]


@pytest.mark.parametrize(
    ("error", "expected_code", "expected_text"),
    [
        (ValueError("bad config"), 1, "Rerun with --verbose"),
        (KeyboardInterrupt(), 130, "interrupted by the user"),
    ],
)
def test_main_reports_failures_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: BaseException,
    expected_code: int,
    expected_text: str,
) -> None:
    import temper.entrypoints.main as entrypoint

    def fail(_config_file):
        raise error

    monkeypatch.setattr(entrypoint, "split_cli", fail)
    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main(["--progress", "off", "split"])

    assert exc_info.value.code == expected_code
    stderr = capsys.readouterr().err
    assert expected_text in stderr
    assert "Traceback" not in stderr


def test_main_verbose_failure_includes_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import temper.entrypoints.main as entrypoint

    def fail(_config_file):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(entrypoint, "split_cli", fail)
    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main(["--verbose", "--progress", "off", "split"])

    assert exc_info.value.code == 1
    stderr = capsys.readouterr().err
    assert "backend exploded" in stderr
    assert "Traceback" in stderr
    assert "Rerun with --verbose" not in stderr


def test_split_cli_loads_config_and_writes_exact_reproduction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import temper.entrypoints.split as entrypoint

    data_root = tmp_path / "data"
    for domain in ("selected", "ignored"):
        domain_path = data_root / domain
        domain_path.mkdir(parents=True)
        (domain_path / "metadata.json").write_text("{}", encoding="utf-8")
    output_root = tmp_path / "split-results"
    config_file = tmp_path / "custom.json"
    requested_trainval_test_seeds = [101, 102]
    requested_train_val_seeds = [201, 202]
    dumpfn(
        {
            "root_path": str(data_root),
            "output_path": str(output_root),
            "domains": None,
            "split_repeats": 2,
            "seed": 7,
            "trainval_test_split_seeds": requested_trainval_test_seeds,
            "train_val_split_seeds": requested_train_val_seeds,
            "test_ratio": 0.2,
            "requested_train_ratios": [0.25, 0.5],
            "max_train_size": 100,
            "train_val_split_method": "random",
            "quests_adapter_config": {
                "descriptor_k": 16,
                "descriptor_cutoff": 4.0,
                "descriptor_dtype": "float32",
                "compute_descriptor_chunk_size": 50,
                "entropy_bandwidth": 0.02,
                "entropy_batch_size": 1000,
                "device": "cpu",
                "gpu_device": None,
                "numba_threads": 2,
            },
            "write_validation": True,
            "write_extra_tests": False,
        },
        config_file,
        indent=2,
    )

    partition_calls: list[tuple] = []
    split_configs = []
    export_calls: list[tuple] = []
    returned_resolvers: list[object] = []
    artifact_dumps: list[tuple[object, Path]] = []
    actual_dumpfn = entrypoint.dumpfn
    reproduce_path = tmp_path / "custom_reproduce.json"

    def partition(domain, root_path, metadata_file_name):
        partition_calls.append((domain, root_path, metadata_file_name))
        return ["grouped-domain"]

    def split(grouped_domain, config):
        split_configs.append(config)
        return ["split-group-1", "split-group-2"]

    def export(split_group, root_path, output_path, **kwargs):
        export_calls.append((split_group, root_path, output_path, kwargs))
        returned_resolver = object()
        returned_resolvers.append(returned_resolver)
        return ["training-unit"], returned_resolver

    def record_dump(obj, path, indent):
        path = Path(path)
        if path == reproduce_path:
            actual_dumpfn(obj, path, indent=indent)
        else:
            artifact_dumps.append((obj, path))

    monkeypatch.setattr(entrypoint, "partition_domain_into_groups", partition)
    monkeypatch.setattr(entrypoint, "split_grouped_domain", split)
    monkeypatch.setattr(entrypoint, "write_all_sets_in_split_group_to_extxyz", export)
    monkeypatch.setattr(entrypoint, "dumpfn", record_dump)

    with caplog.at_level(logging.DEBUG, logger="temper.entrypoints.split"):
        assert entrypoint.split_cli(config_file) == 0

    assert partition_calls == [
        ("ignored", data_root, "metadata.json"),
        ("selected", data_root, "metadata.json"),
    ]
    assert len(split_configs) == 2
    assert split_configs[0].output_path == output_root
    assert split_configs[0].trainval_test_split_seeds == requested_trainval_test_seeds
    assert split_configs[0].train_val_split_seeds == requested_train_val_seeds
    assert export_calls[0][:3] == ("split-group-1", data_root, output_root)
    assert export_calls[0][3]["write_validation"] is True
    assert export_calls[0][3]["write_extra_tests"] is False
    assert export_calls[0][3]["resolver"] is None
    assert export_calls[1][3]["resolver"] is returned_resolvers[0]
    assert export_calls[2][3]["resolver"] is None
    assert export_calls[3][3]["resolver"] is returned_resolvers[2]
    assert [path for _, path in artifact_dumps] == [
        output_root / "ignored" / "grouped_domains.json",
        output_root / "ignored" / "split_groups.json",
        output_root / "ignored" / "training_units.json",
        output_root / "selected" / "grouped_domains.json",
        output_root / "selected" / "split_groups.json",
        output_root / "selected" / "training_units.json",
    ]

    reproduced = loadfn(reproduce_path)
    assert reproduced.domains == ["ignored", "selected"]
    assert reproduced.trainval_test_split_seeds == requested_trainval_test_seeds
    assert reproduced.train_val_split_seeds == requested_train_val_seeds
    assert all(reproduced == config for config in split_configs)
    assert config_file.exists()
    messages = [record.getMessage() for record in caplog.records]
    assert any("Starting split command" in message for message in messages)
    assert any("Resolved split seeds" in message for message in messages)
    assert any("Completed domain" in message for message in messages)


def test_reproduction_config_reloads_and_replays_identical_real_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extxyz_domain: Path,
    metadata_payload: dict,
) -> None:
    import numpy as np

    import temper.entrypoints.split as entrypoint
    import temper.splitting.selectors as selectors_module
    import temper.splitting.split as split_module
    from temper.schemas.split import SplitConfig
    from temper.splitting.quests_adapter import QuestsDescriptorsStorage

    metadata_payload["groupings"] = [{"grouping_strategy": "all"}]
    dumpfn(metadata_payload, extxyz_domain / "metadata.json", indent=2)

    class DeterministicAdapter:
        def __init__(self, config) -> None:
            self.config = config

        def resolve_device(self) -> str:
            return "cpu"

        def compute_descriptors(self, frames: list) -> QuestsDescriptorsStorage:
            frame_count = len(frames)
            return QuestsDescriptorsStorage(
                values=np.arange(frame_count * 2, dtype=float).reshape(frame_count, 2),
                frame_offsets=tuple(range(frame_count + 1)),
                quests_adapter_config=self.config,
            )

        def get_entropy(self, descriptors: np.ndarray) -> float:
            return float(descriptors.sum())

    monkeypatch.setattr(split_module, "QuestsAdapter", DeterministicAdapter)
    monkeypatch.setattr(selectors_module, "QuestsAdapter", DeterministicAdapter)

    output_root = tmp_path / "split-results"
    config_file = tmp_path / "split_config.json"
    dumpfn(
        {
            "root_path": str(extxyz_domain.parent),
            "output_path": str(output_root),
            "domains": None,
            "split_repeats": 2,
            "seed": 11,
            "trainval_test_split_seeds": None,
            "train_val_split_seeds": None,
            "test_ratio": 0.25,
            "requested_train_ratios": [0.5],
            "max_train_size": 2,
            "train_val_split_method": "random",
            "quests_adapter_config": {"device": "cpu"},
            "write_validation": True,
            "write_extra_tests": False,
        },
        config_file,
        indent=2,
    )

    assert entrypoint.split_cli(config_file) == 0

    reproduce_path = tmp_path / "split_config_reproduce.json"
    reproduced = entrypoint._load_split_config(reproduce_path)
    assert isinstance(reproduced, SplitConfig)
    assert reproduced.domains == ["demo_domain"]
    assert reproduced.resolve_domains() is reproduced

    domain_output = output_root / "demo_domain"
    first_split_groups = loadfn(domain_output / "split_groups.json")
    first_training_units = loadfn(domain_output / "training_units.json")
    first_exports = {
        path.name: path.read_bytes()
        for path in sorted(domain_output.glob("*.extxyz"))
    }

    assert entrypoint.split_cli(reproduce_path) == 0

    assert loadfn(domain_output / "split_groups.json") == first_split_groups
    assert loadfn(domain_output / "training_units.json") == first_training_units
    assert {
        path.name: path.read_bytes()
        for path in sorted(domain_output.glob("*.extxyz"))
    } == first_exports

    replayed_reproduce_path = tmp_path / "split_config_reproduce_reproduce.json"
    replayed_reproduction = entrypoint._load_split_config(replayed_reproduce_path)
    assert isinstance(replayed_reproduction, SplitConfig)
    assert replayed_reproduction == reproduced


@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
def test_yaml_split_configs_are_supported_and_preserve_seeds(
    tmp_path: Path,
    suffix: str,
) -> None:
    from temper.entrypoints.split import _load_split_config

    config_file = tmp_path / f"split_config{suffix}"
    dumpfn(
        {
            "split_repeats": 2,
            "seed": 42,
            "trainval_test_split_seeds": [10, 11],
            "train_val_split_seeds": [12, 13],
        },
        config_file,
    )
    config = _load_split_config(config_file)
    assert config.trainval_test_split_seeds == [10, 11]
    assert config.train_val_split_seeds == [12, 13]


def test_reproduction_name_never_matches_input_name() -> None:
    from temper.entrypoints.split import _load_split_config, _reproduce_config_path

    assert _reproduce_config_path(Path("split_config.json")) == Path(
        "split_config_reproduce.json"
    )
    assert _reproduce_config_path(Path("split_config_reproduce.json")) == Path(
        "split_config_reproduce_reproduce.json"
    )
    with pytest.raises(ValueError, match="JSON or YAML"):
        _load_split_config(Path("split_config.toml"))
