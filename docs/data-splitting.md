# Data splitting

[Back to the project overview](../README.md) · [Data format and grouping](raw-data-format.md) · [Roadmap](roadmap.md)

TEMPER splits a [`GroupedDomain`](../src/temper/schemas/group.py) into reference-based [`SplitGroup`](../src/temper/schemas/split.py) results. Use the end-to-end CLI for the standard workflow or the Python API for finer control.

## Command-line workflow

Copy or adapt the complete [default configuration example](split_config.example.json)
to `split_config.json`, then run:

```console
python -m src.temper.entrypoints.main split
```

The split subcommand accepts no individual split parameters. To use another JSON or
YAML file, provide its path as the sole configuration selector:

```console
python -m src.temper.entrypoints.main split --config-file configs/sse_llzo.yaml
```

### Logging and live progress

Logging options are global main-command options, not split options. Place them
before the subcommand so the same interface can be reused by future TEMPER
commands:

```console
temper_bench --verbose split --config-file configs/sse_llzo.yaml
temper_bench --log-level WARNING split
temper_bench --progress plain split 2> split.log
```

The available controls are:

| Option | Behavior |
| --- | --- |
| default / `--log-level INFO` | Major command, domain, grouping, splitting, and export phases plus live progress. |
| `-v`, `--verbose` | `DEBUG` diagnostics including paths, seeds, descriptor chunks, selector checkpoints, exports, timings, and failure tracebacks. |
| `-q`, `--quiet` | `WARNING` and `ERROR` records only. |
| `--log-level LEVEL` | Explicit `DEBUG`, `INFO`, `WARNING`, or `ERROR` threshold. |
| `--progress auto` | Default: one reusable status line on a TTY, or plain heartbeat records otherwise. |
| `--progress plain` | Always emit newline-based heartbeats, suitable for notebooks and CI. |
| `--progress off` | Disable status and heartbeat output without suppressing lifecycle logs. |

All diagnostics go to stderr. When stderr is redirected, TEMPER never writes
an animated progress bar or one line per progress update. Instead, an active
long operation emits at most one `Still ...` heartbeat per minute, including
its current count, context, and elapsed time. Frequent inner-loop details remain
available at `DEBUG`.

Python API users retain control of application logging. To receive the same
module records without invoking the CLI, configure Python logging before
calling TEMPER:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

TEMPER warning categories are exported from `temper.logging`. They derive from
`TemperWarning`, with specific categories for data quality, backend fallback,
and performance conditions, so API users may filter them through Python's
standard `warnings` module:

```python
import warnings

from temper.logging import TemperWarning

warnings.filterwarnings("error", category=TemperWarning)
```

When `--config-file` is omitted, the CLI reads the path in
`DEFAULT_SPLIT_CONFIG_FILE`; that environment variable defaults to
`split_config.json`. Relative paths in the configuration are interpreted relative to
the process's current working directory.

With `"domains": null`, the command discovers the direct child directories of
`root_path` that contain `metadata.json`. Discovery is performed explicitly during
command startup, and the resulting domain names are sorted and stored in the resolved
configuration. Set `domains` to a list such as `["sse_llzo"]` to select specific
domain directory names without filesystem discovery. The CLI groups and splits each
selected domain, exports its datasets, and writes:

```text
split_results/
└── sse_llzo/
    ├── grouped_domains.json
    ├── split_groups.json
    ├── training_units.json
    └── *.extxyz
```

Set `write_validation` to `true` to materialize validation files and
`write_extra_tests` to `false` to omit cross-test datasets.

### Reproduction file and exact seed replay

After loading and validating a configuration, `split_cli` resolves automatic domain
discovery and writes the resulting MSON-compatible JSON configuration beside the
input. The filename replaces the input extension with `_reproduce.json`; for example:

- `split_config.json` writes `split_config_reproduce.json`;
- `configs/run.yaml` writes `configs/run_reproduce.json`; and
- replaying `split_config_reproduce.json` writes
  `split_config_reproduce_reproduce.json`.

The input file is therefore never overwritten. The reproduction file contains the
concrete `domains` used by the run as well as the resolved `seed`,
`trainval_test_split_seeds`, and `train_val_split_seeds`. Pass that file back to
`--config-file` to reuse the same domain scope and both per-repeat seed lists exactly:

```console
python -m src.temper.entrypoints.main split \
  --config-file split_config_reproduce.json
```

For a fresh run, set the two per-repeat seed lists to `null`. A fixed nonnegative
`seed` deterministically derives them; `seed: null` generates and persists a new base
seed. For an exact replay, provide both lists (as the reproduction file does). Their
lengths must equal `split_repeats`; supplied values are validated and used unchanged,
not regenerated from `seed`.

Likewise, rerunning an original configuration with `"domains": null` discovers the
current domain directories again. Replaying its reproduction file uses the explicit
domain list captured by the earlier run instead.

## Public workflow

```python
from src.temper.grouping import partition_domain_into_groups
from src.temper.splitting import (
    split_grouped_domain,
    write_all_sets_in_split_group_to_extxyz,
)
from src.temper.schemas.quests_adapter import QuestsAdapterConfig
from src.temper.schemas.split import SplitConfig

# Load one GroupedDomain for each grouping specification in metadata.json.
grouped_domains = partition_domain_into_groups("sse_llzo", root_path="./data")

config = SplitConfig(
    root_path="./data",
    output_path="./split_results",
    split_repeats=3,
    seed=11,
    train_val_split_method="quests",
    quests_adapter_config=QuestsAdapterConfig(device="cpu"),
)

split_groups = split_grouped_domain(grouped_domains[0], config)

# Extra tests are enabled by default for this exporter, so provide all results.
training_units, resolver = write_all_sets_in_split_group_to_extxyz(
    split_groups[0],
    root_path=config.root_path,
    output_path=config.output_path,
    all_split_groups=split_groups,
)
```

[`partition_domain_into_groups`](../src/temper/grouping/group.py) reads the domain metadata and produces one grouped domain per grouping definition. [`split_grouped_domain`](../src/temper/splitting/split.py) splits **every group** in its input for **every repeat**, returning a flat `list[SplitGroup]`. A [`SplitConfig`](../src/temper/schemas/split.py) selects one train/validation method per call: `random` or `quests`.

## Split configuration and defaults

`SplitConfig` accepts:

- `root_path`: source data root, defaulting to `DEFAULT_DATA_DIR`;
- `output_path`: results root, defaulting to `DEFAULT_SPLIT_RESULTS_DIR`;
- `domains`: domain names to process, or `null` to discover domains from metadata;
- `split_repeats`: number of repeats;
- `seed`: nonnegative base seed from which both per-repeat seed lists are derived;
- `trainval_test_split_seeds` and `train_val_split_seeds`: persisted per-repeat seeds, normally generated rather than supplied directly;
- `test_ratio`: fraction of each group held out for its own test set;
- `requested_train_ratios`: nested training sizes as fractions of the train+validation pool;
- `max_train_size`: cap on the largest requested training set;
- `train_val_split_method`: `random` or `quests`; and
- `quests_adapter_config`: a [`QuestsAdapterConfig`](../src/temper/splitting/quests_adapter.py);
- `write_validation`: whether validation datasets are exported; and
- `write_extra_tests`: whether configured cross-test datasets are exported.

Defaults are defined in [`src/temper/utils/defaults.py`](../src/temper/utils/defaults.py): the default configuration file is `split_config.json`, data root is `./data`, split-results root is `./split_results`, metadata filename is `metadata.json`, test ratio is `0.2`, training ratios are `[0.1, 0.2, 0.4, 0.6, 0.8, 0.9]`, maximum training size is `3000`, and repeat count is `3`. Environment variables with the corresponding default names are read when that module is imported.

The [complete JSON example](split_config.example.json) lists every option with its
built-in default when no environment overrides are set. JSON and YAML use the same
field names and nesting. `numba_threads: null` selects the dynamic default of half the
available CPU count, capped at 8; the reproduction file records the resulting value.

Constructing `SplitConfig` with a base `seed` deterministically derives two seed lists of length `split_repeats`: one for train+validation/test partitioning and one for train/validation selection. If `seed` is omitted or `None`, a nonnegative base seed is generated and stored in the model so the resulting configuration remains reproducible. Explicit seed lists always take precedence and are preserved exactly.

## What each split does

For each repeat and each group, the implementation:

1. builds the group's ordered [`FrameReference`](../src/temper/schemas/split.py:34) pool;
2. randomly chooses `round(pool_size * test_ratio)` references for the group's test set using the corresponding `trainval_test_split_seeds` value;
3. computes QUESTS descriptors for the whole group;
4. selects the requested nested training prefixes from the remaining train+validation pool with the configured `random` or `quests` selector; and
5. returns a [`SplitGroup`](../src/temper/schemas/split.py:318) containing the test references, one [`TrainValSplitTrajectory`](../src/temper/schemas/split.py:174), provenance, and any assigned extra test groups.

The test ratio must produce at least one test frame and at least one train+validation frame. Requested ratios are converted to integer sizes and reduced proportionally if their largest size would exceed `max_train_size`.

At checkpoint index `i`, [`TrainValSplitTrajectory.get_train_set`](../src/temper/schemas/split.py:275) returns the prefix of `selected_frames` with `requested_train_sizes[i]` references. [`TrainValSplitTrajectory.get_val_set`](../src/temper/schemas/split.py:293) returns the remaining selected suffix followed by `additional_trainval_frames`. The trajectory therefore stores the entire train+validation inventory by reference, while its training sets are nested prefixes.

## Persistence and reconstruction

[`SplitGroup`](../src/temper/schemas/split.py), [`SplitConfig`](../src/temper/schemas/split.py), and the other persisted schemas derive from [`MSONableModel`](../src/temper/schemas/base.py). Use Monty's `dumpfn` and `loadfn` to write and reconstruct them. A split result persists references, not `ase.Atoms` objects or computed descriptors.

```python
from monty.serialization import dumpfn, loadfn

dumpfn(split_groups, "split_groups.json", indent=2)
restored_split_groups = loadfn("split_groups.json")
```

`SplitConfig` serializes both `root_path` and `output_path` as portable strings and
reconstructs them as `Path` values. The CLI also accepts a plain JSON/YAML mapping
without Monty's `@module` and `@class` metadata.

A [`FrameReferenceResolver`](../src/temper/splitting/io.py:87) resolves each reference as `root_path / domain / filename`, protects against traversal outside that location, reads each source file at most once per resolver lifetime, and validates energy and forces. Treat frames returned through its cache as read-only.

Persisted `SplitGroup` records include a deterministic UUIDv5 `split_id`. It is
derived from the materialized split definition rather than randomly generated.
The ID is stored and verified once when loaded, so ordinary access does not
rehash the frame-reference collections. Validated reassignment of a split field
regenerates it; the derived entropy profile is excluded from identity.

The lower-level reconstruction helpers are [`load_frames_from_references`](../src/temper/splitting/io.py:283), [`load_frames_test`](../src/temper/splitting/io.py:335), and [`load_frames_train_validation`](../src/temper/splitting/io.py:379). They preserve reference order and return both reconstructed frames and the resolver used.

## Export

[`write_all_sets_in_split_group_to_extxyz`](../src/temper/splitting/io.py:608) reconstructs and writes every requested training checkpoint plus the group's own test set. It returns a list of [`TrainingUnit`](../src/temper/schemas/train_unit.py:9) records and the reusable resolver.

- Exported names are deterministic and include domain, grouping strategy, group, method, role, frame count, and repeat ID.
- Dataset files are written under `output_path / split_group.domain`; writes atomically replace existing generated files.
- `write_validation=False` by default; set it to `True` to write non-empty validation datasets.
- `write_extra_tests=True` by default. In that mode, `all_split_groups` is required so the exporter can find and write the test sets of groups named in `extra_tested_groups`.
- Extra test files are recorded alongside the ordinary test file in each returned `TrainingUnit`; they are not added to `SplitGroup.test_set` itself.
- Each `TrainingUnit.root_path` is the shared results root, and its file validation resolves datasets beneath `root_path / domain`.
- Each newly exported unit records its parent `split_id` and stores a deterministic UUIDv5 `training_unit_id`. Validated reassignment regenerates that ID, while relocating the excluded `root_path` leaves it unchanged.

## QUESTS backend

QUESTS is a required dependency of this project and provides descriptor and entropy calculations for both supported selection paths. [`QuestsAdapterConfig`](../src/temper/splitting/quests_adapter.py:39) controls descriptor settings, entropy settings, CPU thread count, and backend routing.

The base installation does not install PyTorch. Because ordinary pip dependency
metadata cannot select a hardware-specific PyTorch package index, install a
compatible PyTorch build before TEMPER. Use the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/) to
choose a CUDA build compatible with an NVIDIA GPU and driver, or a supported
ROCm build for an AMD GPU. Then verify the installation:

```console
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip, torch.cuda.is_available())"
python -m pip install .
```

The final value printed by the first command must be `True`, but that is only an
availability check. It confirms device discovery and does not prove that the
installed PyTorch build contains kernels for the GPU architecture.
Force a small kernel launch as a smoke test:

```console
python -c "import torch; x = torch.ones(1, device='cuda'); print((x + 1).cpu())"
```

On NVIDIA systems, inspect both the device capability and the architectures in
the installed PyTorch build:

```console
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"
```

PyTorch's ROCm build intentionally uses the `torch.cuda` API and `cuda` device
spelling too.

Do not assume that `pip install torch` detects the GPU. As documented in the
[PyTorch 2.13 release notes](https://pytorch.org/blog/pytorch-2-13-release-blog/),
the default PyPI build on Linux and Windows uses CUDA 13.0. CUDA 13
[does not support Maxwell, Pascal, or Volta](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#deprecated-architectures);
these GPUs need a compatible CUDA 12.x build (PyTorch 2.13 provides CUDA 12.6).
AMD GPUs need an appropriate ROCm build. Consult the selector whenever
installing because the default and compatibility matrix can change between
PyTorch releases.

A V100 is a Volta GPU with compute capability 7.0. An incompatible PyTorch
build can therefore report `torch.cuda.is_available() == True` but fail at the
first real operation with `CUDA error: no kernel image is available for
execution on the device`.

If the default PyPI PyTorch build is known to be suitable, the optional `gpu`
extra is a convenience:

```console
python -m pip install ".[gpu]"
```

For an installed distribution, use
`python -m pip install "temper-bench[gpu]"`. The extra installs the default
`torch` package but cannot choose a custom CUDA or ROCm package index.

The `device` option controls routing:

- `cpu` always uses the CPU and does not initialize torch or a GPU runtime;
- `auto` (the default) uses the GPU when `torch.cuda.is_available()` is true
  and otherwise falls back to the CPU; and
- `gpu` requires the GPU route and raises an error instead of falling back when
  torch or a compatible accelerator is unavailable.

`auto` is availability-based routing, not a kernel-compatibility preflight. It
selects the GPU solely when `torch.cuda.is_available()` is true. TEMPER does not
launch a probe kernel during routing and does not fall back to CPU if a later
PyTorch/QUESTS kernel launch fails. In that case, install a PyTorch build that
supports the GPU architecture or set `device` to `cpu`.

Set `gpu_device` to a PyTorch device such as `cuda:0` when a specific device is
required. PyTorch uses this `cuda` spelling for ROCm devices as well. The
resolved configuration is stored as split provenance. For example, this
configuration requires the first GPU device:

```json
{
  "quests_adapter_config": {
    "device": "gpu",
    "gpu_device": "cuda:0"
  }
}
```

## Cross-test assignments

Each split result records `extra_tested_groups`. If the input grouped domain has `specify_cross_tests`, the splitter uses that explicit mapping regardless of `add_extra_cross_tests`, after removing each source group's self-reference and deduplicating its targets. Otherwise, when `add_extra_cross_tests` is `True`, every group is assigned every other group as an extra test group; when it is `False`, no extra cross tests are assigned. The automatically generated other-group target ordering is not guaranteed because it is derived from sets. Extra group datasets remain separate from `SplitGroup.test_set` and are reconstructed for export through `extra_tested_groups`. See [Data format and grouping](raw-data-format.md#implemented-cross-test-behavior) for the metadata details.
