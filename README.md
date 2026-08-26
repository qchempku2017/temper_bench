# TEMPER benchmark

TEMPER provides the implemented data-preparation layer for a benchmark of machine-learned force fields (MLFFs). It reads labeled `extxyz` data, organizes each domain into groups, creates reproducible train/validation/test splits, and exports referenced frames back to `extxyz` files.

## What is implemented

The workflow is available through both the CLI and Python API:

1. Prepare a domain directory containing labeled `extxyz` files and a `metadata.json` file.
2. Call [`partition_domain_into_groups`](src/temper/grouping/group.py) to load the metadata and produce one grouped domain per configured grouping strategy.
3. Call [`split_grouped_domain`](src/temper/splitting/split.py) with a [`SplitConfig`](src/temper/schemas/split.py) to split every group for every configured repeat.
4. Persist models with Monty serialization, or reconstruct and export datasets with [`FrameReferenceResolver`](src/temper/splitting/io.py) and [`write_all_sets_in_split_group_to_extxyz`](src/temper/splitting/io.py).

The end-to-end command reads every option from a JSON or YAML [`SplitConfig`](docs/split_config.example.json). By default it reads `split_config.json` from the current directory:

Install TEMPER from the repository and invoke its command-line entry point:

```console
python -m pip install .
temper_bench split
```

### Optional GPU support

The base installation supports the QUESTS CPU backend and does not install
PyTorch. For GPU execution, the recommended installation order is:

1. Use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/)
   to choose a build compatible with the operating system, GPU, and driver.
   NVIDIA users should check the CUDA version and GPU architecture; AMD users
   need a supported ROCm build.
2. Install that PyTorch build and verify that it can see the accelerator:

   ```console
   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip, torch.cuda.is_available())"
   ```

   The final value must be `True` for the QUESTS GPU route.
3. Install TEMPER without the extra, so pip leaves the selected PyTorch build
   alone:

   ```console
   python -m pip install .
   ```

> [!WARNING]
> Ordinary `pip install torch` does not inspect the installed GPU and choose a
> matching build. As documented in the
> [PyTorch 2.13 release notes](https://pytorch.org/blog/pytorch-2-13-release-blog/),
> the default PyPI build on Linux and Windows uses CUDA 13.0. CUDA 13
> [removed Maxwell, Pascal, and Volta support](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#deprecated-architectures);
> those NVIDIA GPUs need a compatible CUDA 12.x build (PyTorch 2.13 provides a
> CUDA 12.6 build). AMD GPUs need a compatible ROCm build rather than the
> default CUDA build. Recheck the selector and compatibility information when
> installing because PyTorch's defaults change between releases.

For systems where the current default PyPI PyTorch build is known to be
compatible, the `gpu` extra remains a convenience:

```console
python -m pip install ".[gpu]"
```

The extra cannot select a custom CUDA or ROCm package index. Leave
`quests_adapter_config.device` as `auto` to use a PyTorch GPU when available
and fall back to the CPU otherwise, or set it to `gpu` to require the GPU and
fail if it is unavailable. See [Data splitting](docs/data-splitting.md#quests-backend)
for the complete backend configuration.

The module form is also available as `python -m temper.entrypoints.main split`.

Logging and live-progress controls belong to the main `temper_bench` command and
therefore come before the subcommand. The default `INFO` level reports major
phases and keeps long work visible without printing per-frame details:

```console
temper_bench --verbose split --config-file path/to/custom.yaml
temper_bench --quiet split
temper_bench --log-level WARNING --progress plain split 2> temper.log
```

`--verbose` enables developer diagnostics and tracebacks; `--quiet` shows only
warnings and errors. `--log-level` accepts `DEBUG`, `INFO`, `WARNING`, or
`ERROR`. Progress defaults to one reusable status line on an interactive
terminal and one heartbeat per minute when stderr is redirected. Use
`--progress plain` to force heartbeat lines or `--progress off` to disable live
status while keeping ordinary lifecycle logs.

Use `--config-file path/to/custom.yaml` to select another file, or set the
`DEFAULT_SPLIT_CONFIG_FILE` environment variable. The command writes a resolved
`<config-stem>_reproduce.json` next to the input configuration, including the concrete
domain list and exact generated or supplied seeds needed to replay the split.
Configuration parameters are explained in [`data splitting`](docs/data-splitting.md)

Each domain receives `grouped_domains.json`, `split_groups.json`, `training_units.json`, and its generated `extxyz` datasets under `split_results/<domain>/`.

Split records store frame references rather than embedded structures or descriptors. A [`FrameReference`](src/temper/schemas/split.py:34) identifies one source frame by domain, relative `extxyz` filename, and zero-based frame index. This keeps the split compact and lets it be reconstructed from the original data tree.

## Current scope

Grouping, splitting, QUESTS-backed selection, split persistence, frame reconstruction, `extxyz` export, and the end-to-end split CLI are implemented. Training jobs, benchmark execution, result uploading, and metrics are not implemented features.

## Documentation

- [Data format and grouping](docs/raw-data-format.md) — required domain layout and metadata, inventory autodetection, grouping strategies, and cross-test behavior.
- [Data splitting](docs/data-splitting.md) — split configuration, result schemas, Python API, QUESTS configuration, reference-based reconstruction, and export.
- [Roadmap](docs/roadmap.md) — clearly marked planned capabilities.
