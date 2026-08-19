"""Tests for the top-level CLI parser and split orchestration."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_main_parser_exposes_split_refactor_options(tmp_path: Path) -> None:
    from src.temper.entrypoints.main import main_parser

    args = main_parser().parse_args([
        "split",
        "--root-path", str(tmp_path / "data"),
        "--output-path", str(tmp_path / "results"),
        "--domains", "first", "second",
        "--method", "random",
        "--max-train-size", "125",
        "--device", "auto",
    ])

    assert args.command == "split"
    assert args.domains == ["first", "second"]
    assert args.max_train_size == 125
    assert args.method == "random"
    assert args.device == "auto"


def test_split_cli_filters_domains_and_writes_per_domain_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.temper.entrypoints.split as entrypoint

    data_root = tmp_path / "data"
    for domain in ("selected", "ignored"):
        domain_path = data_root / domain
        domain_path.mkdir(parents=True)
        (domain_path / "metadata.json").write_text("{}", encoding="utf-8")
    output_root = tmp_path / "split-results"

    partition_calls: list[tuple] = []
    split_configs = []
    export_calls: list[tuple] = []
    dumps: list[tuple[object, Path]] = []

    def partition(domain, root_path, metadata_file_name):
        partition_calls.append((domain, root_path, metadata_file_name))
        return ["grouped-domain"]

    def split(grouped_domain, config):
        split_configs.append(config)
        return ["split-group"]

    def export(split_group, root_path, output_path, **kwargs):
        export_calls.append((split_group, root_path, output_path, kwargs))
        return ["training-unit"], None

    monkeypatch.setattr(entrypoint, "partition_domain_into_groups", partition)
    monkeypatch.setattr(entrypoint, "split_grouped_domain", split)
    monkeypatch.setattr(entrypoint, "write_all_sets_in_split_group_to_extxyz", export)
    monkeypatch.setattr(entrypoint, "dumpfn", lambda obj, path, indent: dumps.append((obj, Path(path))))

    result = entrypoint.split_cli(
        root_path=data_root,
        output_path=output_root,
        domains=["selected"],
        split_repeats=2,
        seed=7,
        test_ratio=0.2,
        train_ratios=[0.25, 0.5],
        max_train_size=100,
        method="random",
        descriptor_k=16,
        descriptor_cutoff=4.0,
        descriptor_dtype="float32",
        compute_descriptor_chunk_size=50,
        entropy_bandwidth=0.02,
        entropy_batch_size=1000,
        device="cpu",
        gpu_device=None,
        numba_threads=2,
        write_validation=True,
        write_extra_tests=False,
    )

    assert result == 0
    assert partition_calls == [("selected", data_root, "metadata.json")]
    assert len(split_configs) == 1
    assert split_configs[0].seed == 7
    assert len(split_configs[0].trainval_test_split_seeds) == 2
    assert len(split_configs[0].train_val_split_seeds) == 2
    assert export_calls[0][:3] == ("split-group", data_root, output_root)
    assert export_calls[0][3]["write_validation"] is True
    assert export_calls[0][3]["write_extra_tests"] is False
    assert [path for _, path in dumps] == [
        output_root / "selected" / "grouped_domains.json",
        output_root / "selected" / "split_groups.json",
        output_root / "selected" / "training_units.json",
    ]
    assert (output_root / "selected").is_dir()
