"""Tests for the top-level CLI parser and split orchestration."""
from __future__ import annotations

from pathlib import Path

from monty.serialization import dumpfn, loadfn
import pytest


def test_main_parser_accepts_only_a_split_config_file() -> None:
    from temper.entrypoints.main import main_parser
    from temper.utils.defaults import DEFAULT_SPLIT_CONFIG_FILE

    default_args = main_parser().parse_args(["split"])
    assert default_args.command == "split"
    assert default_args.config_file == Path(DEFAULT_SPLIT_CONFIG_FILE)

    args = main_parser().parse_args(["split", "--config-file", "custom.yaml"])
    assert args.config_file == Path("custom.yaml")
    with pytest.raises(SystemExit):
        main_parser().parse_args(["split", "--root-path", "data"])


def test_split_cli_loads_config_and_writes_exact_reproduction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
            "domains": ["selected"],
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
    artifact_dumps: list[tuple[object, Path]] = []
    actual_dumpfn = entrypoint.dumpfn
    reproduce_path = tmp_path / "custom_reproduce.json"

    def partition(domain, root_path, metadata_file_name):
        partition_calls.append((domain, root_path, metadata_file_name))
        return ["grouped-domain"]

    def split(grouped_domain, config):
        split_configs.append(config)
        return ["split-group"]

    def export(split_group, root_path, output_path, **kwargs):
        export_calls.append((split_group, root_path, output_path, kwargs))
        return ["training-unit"], None

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

    assert entrypoint.split_cli(config_file) == 0

    assert partition_calls == [("selected", data_root, "metadata.json")]
    assert len(split_configs) == 1
    assert split_configs[0].output_path == output_root
    assert split_configs[0].trainval_test_split_seeds == requested_trainval_test_seeds
    assert split_configs[0].train_val_split_seeds == requested_train_val_seeds
    assert export_calls[0][:3] == ("split-group", data_root, output_root)
    assert export_calls[0][3]["write_validation"] is True
    assert export_calls[0][3]["write_extra_tests"] is False
    assert [path for _, path in artifact_dumps] == [
        output_root / "selected" / "grouped_domains.json",
        output_root / "selected" / "split_groups.json",
        output_root / "selected" / "training_units.json",
    ]

    reproduced = loadfn(reproduce_path)
    assert reproduced.trainval_test_split_seeds == requested_trainval_test_seeds
    assert reproduced.train_val_split_seeds == requested_train_val_seeds
    assert reproduced == split_configs[0]
    assert config_file.exists()


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
