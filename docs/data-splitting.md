# Data splitting

[Back to the project overview](../README.md) · [Data format and grouping](raw-data-format.md) · [Roadmap](roadmap.md)

TEMPER splits a [`GroupedDomain`](../src/temper/schemas/group.py) into reference-based [`SplitGroup`](../src/temper/schemas/split.py) results. Use the end-to-end CLI for the standard workflow or the Python API for finer control.

## Command-line workflow

```console
python -m src.temper.entrypoints.main split \
  --root-path ./data \
  --output-path ./split_results \
  --domains sse_llzo \
  --seed 11 \
  --device cpu
```

Without `--domains`, the command discovers all domain directories beneath `--root-path` that contain `metadata.json`. It groups and splits each selected domain, exports its datasets, and writes:

```text
split_results/
└── sse_llzo/
    ├── grouped_domains.json
    ├── split_groups.json
    ├── training_units.json
    └── *.extxyz
```

Use `--write-validation` to materialize validation files and `--no-write-extra-tests` to omit cross-test datasets. Run the command with `--help` for the split, ratio, size, QUESTS, and device options.

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
    split_repeats=3,
    seed=11,
    train_val_split_method="quests",
    quests_adapter_config=QuestsAdapterConfig(device="cpu"),
)

split_groups = split_grouped_domain(grouped_domains[0], config)

# Extra tests are enabled by default for this exporter, so provide all results.
training_units, resolver = write_all_sets_in_split_group_to_extxyz(
    split_groups[0],
    root_path="./data",
    output_path="./split_results",
    all_split_groups=split_groups,
)
```

[`partition_domain_into_groups`](../src/temper/grouping/group.py) reads the domain metadata and produces one grouped domain per grouping definition. [`split_grouped_domain`](../src/temper/splitting/split.py) splits **every group** in its input for **every repeat**, returning a flat `list[SplitGroup]`. A [`SplitConfig`](../src/temper/schemas/split.py) selects one train/validation method per call: `random` or `quests`.

## Split configuration and defaults

`SplitConfig` accepts:

- `root_path`: source data root, defaulting to `DEFAULT_DATA_DIR`;
- `split_repeats`: number of repeats;
- `seed`: nonnegative base seed from which both per-repeat seed lists are derived;
- `trainval_test_split_seeds` and `train_val_split_seeds`: persisted per-repeat seeds, normally generated rather than supplied directly;
- `test_ratio`: fraction of each group held out for its own test set;
- `requested_train_ratios`: nested training sizes as fractions of the train+validation pool;
- `max_train_size`: cap on the largest requested training set;
- `train_val_split_method`: `random` or `quests`; and
- `quests_adapter_config`: a [`QuestsAdapterConfig`](../src/temper/splitting/quests_adapter.py:39).

Defaults are defined in [`src/temper/utils/defaults.py`](../src/temper/utils/defaults.py): the default data root is `./data`, split-results root is `./split_results`, metadata filename is `metadata.json`, test ratio is `0.2`, training ratios are `[0.1, 0.2, 0.4, 0.6, 0.8, 0.9]`, maximum training size is `3000`, and repeat count is `3`. Environment variables with the corresponding default names are read when that module is imported.

Constructing `SplitConfig` with a base `seed` deterministically derives two seed lists of length `split_repeats`: one for train+validation/test partitioning and one for train/validation selection. If `seed` is omitted or `None`, a nonnegative base seed is generated and stored in the model so the resulting configuration remains reproducible.

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

A [`FrameReferenceResolver`](../src/temper/splitting/io.py:87) resolves each reference as `root_path / domain / filename`, protects against traversal outside that location, reads each source file at most once per resolver lifetime, and validates energy and forces. Treat frames returned through its cache as read-only.

The lower-level reconstruction helpers are [`load_frames_from_references`](../src/temper/splitting/io.py:283), [`load_frames_test`](../src/temper/splitting/io.py:335), and [`load_frames_train_validation`](../src/temper/splitting/io.py:379). They preserve reference order and return both reconstructed frames and the resolver used.

## Export

[`write_all_sets_in_split_group_to_extxyz`](../src/temper/splitting/io.py:608) reconstructs and writes every requested training checkpoint plus the group's own test set. It returns a list of [`TrainingUnit`](../src/temper/schemas/train_unit.py:9) records and the reusable resolver.

- Exported names are deterministic and include domain, grouping strategy, group, method, role, frame count, and repeat ID.
- Dataset files are written under `output_path / split_group.domain`; writes atomically replace existing generated files.
- `write_validation=False` by default; set it to `True` to write non-empty validation datasets.
- `write_extra_tests=True` by default. In that mode, `all_split_groups` is required so the exporter can find and write the test sets of groups named in `extra_tested_groups`.
- Extra test files are recorded alongside the ordinary test file in each returned `TrainingUnit`; they are not added to `SplitGroup.test_set` itself.
- Each `TrainingUnit.root_path` is the shared results root, and its file validation resolves datasets beneath `root_path / domain`.

## QUESTS backend

QUESTS is a required dependency of this project and provides descriptor and entropy calculations for both supported selection paths. [`QuestsAdapterConfig`](../src/temper/splitting/quests_adapter.py:39) controls descriptor settings, entropy settings, CPU thread count, and backend routing.

Its `device` option is `auto` by default. `cpu` keeps execution on the CPU and does not initialize torch or CUDA. `gpu` requires a usable CUDA device and torch; `auto` uses GPU when available and otherwise falls back to CPU. GPU support is optional and is not installed by the base requirements. The configuration is stored as split provenance.

## Cross-test assignments

Each split result records `extra_tested_groups`. If the input grouped domain has `specify_cross_tests`, the splitter uses that explicit mapping regardless of `add_extra_cross_tests`, after removing each source group's self-reference and deduplicating its targets. Otherwise, when `add_extra_cross_tests` is `True`, every group is assigned every other group as an extra test group; when it is `False`, no extra cross tests are assigned. The automatically generated other-group target ordering is not guaranteed because it is derived from sets. Extra group datasets remain separate from `SplitGroup.test_set` and are reconstructed for export through `extra_tested_groups`. See [Data format and grouping](raw-data-format.md#implemented-cross-test-behavior) for the metadata details.
