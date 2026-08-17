# Data splitting

[Back to the project overview](../README.md) · [Data format and grouping](raw-data-format.md) · [Roadmap](roadmap.md)

TEMPER splits a [`GroupedDomain`](../src/temper/schemas/group.py:15) into reference-based [`SplitGroup`](../src/temper/schemas/split.py:318) results. The implementation has no CLI: use the Python API below.

## Public workflow

```python
from temper.grouping import partition_domain_groups
from temper.splitting import (
    QuestsAdapterConfig,
    split_grouped_domain,
    write_all_sets_in_split_group_to_extxyz,
)
from src.temper.splitting.split import SplitConfig

# Load one GroupedDomain for each grouping specification in metadata.json.
grouped_domains = partition_domain_groups("sse_llzo", root_path="./data")

config = SplitConfig(
    root_path="./data",
    split_repeats=3,
    trainval_test_split_seeds=[11, 12, 13],
    train_val_split_seeds=[21, 22, 23],
    train_val_split_method="quests",
    quests_adapter_config=QuestsAdapterConfig(device="cpu"),
)

split_groups = split_grouped_domain(grouped_domains[0], config)

# Extra tests are enabled by default for this exporter, so provide all results.
training_units, resolver = write_all_sets_in_split_group_to_extxyz(
    split_groups[0],
    root_path="./data",
    output_path="./train_units",
    all_split_groups=split_groups,
)
```

[`partition_domain_groups`](../src/temper/grouping/group.py:15) reads the domain metadata and produces one grouped domain per grouping definition. [`split_grouped_domain`](../src/temper/splitting/split.py:237) splits **every group** in its input for **every repeat**, returning a flat `list[SplitGroup]`. A [`SplitConfig`](../src/temper/splitting/split.py:128) selects one train/validation method per call: `random` or `quests`.

## Split configuration and defaults

`SplitConfig` accepts:

- `root_path`: source data root, defaulting to `DEFAULT_DATA_DIR`;
- `split_repeats`: number of repeats;
- `trainval_test_split_seeds`: one seed per repeat for the random train+validation/test partition;
- `train_val_split_seeds`: one seed per repeat for train/validation selection;
- `test_ratio`: fraction of each group held out for its own test set;
- `requested_train_ratios`: nested training sizes as fractions of the train+validation pool;
- `max_train_size`: cap on the largest requested training set;
- `train_val_split_method`: `random` or `quests`; and
- `quests_adapter_config`: a [`QuestsAdapterConfig`](../src/temper/splitting/quests_adapter.py:39).

Defaults are defined in [`src/temper/utils/defaults.py`](../src/temper/utils/defaults.py): the default data root is `./data`, metadata filename is `metadata.json`, test ratio is `0.2`, training ratios are `[0.1, 0.2, 0.4, 0.6, 0.8, 0.9]`, maximum training size is `3000`, and repeat count is `3`. Environment variables with the corresponding default names are read when that module is imported.

When `SplitConfig()` is constructed without explicitly passing seed lists, its current default factories create empty lists. Provide seed lists with exactly `split_repeats` nonnegative integers for a usable, reproducible configuration. Passing `None` explicitly instead generates random seed values during configuration validation.

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

[`SplitGroup`](../src/temper/schemas/split.py:318), [`SplitConfig`](../src/temper/splitting/split.py:128), and the other JSON models inherit JSON save/load helpers from [`JsonIOModel`](../src/temper/schemas/base.py:1). A split result persists references, not `ase.Atoms` objects or computed descriptors.

A [`FrameReferenceResolver`](../src/temper/splitting/io.py:87) resolves each reference as `root_path / domain / filename`, protects against traversal outside that location, reads each source file at most once per resolver lifetime, and validates energy and forces. Treat frames returned through its cache as read-only.

The lower-level reconstruction helpers are [`load_frames_from_references`](../src/temper/splitting/io.py:283), [`load_frames_test`](../src/temper/splitting/io.py:335), and [`load_frames_train_validation`](../src/temper/splitting/io.py:379). They preserve reference order and return both reconstructed frames and the resolver used.

## Export

[`write_all_sets_in_split_group_to_extxyz`](../src/temper/splitting/io.py:608) reconstructs and writes every requested training checkpoint plus the group's own test set. It returns a list of [`TrainingUnit`](../src/temper/schemas/train_unit.py:9) records and the reusable resolver.

- Exported names are deterministic and include domain, grouping strategy, group, method, role, frame count, and repeat ID.
- Writes atomically replace existing generated files in `output_path`.
- `write_validation=False` by default; set it to `True` to write non-empty validation datasets.
- `write_extra_tests=True` by default. In that mode, `all_split_groups` is required so the exporter can find and write the test sets of groups named in `extra_tested_groups`.
- Extra test files are recorded alongside the ordinary test file in each returned `TrainingUnit`; they are not added to `SplitGroup.test_set` itself.

## QUESTS backend

QUESTS is a required dependency of this project and provides descriptor and entropy calculations for both supported selection paths. [`QuestsAdapterConfig`](../src/temper/splitting/quests_adapter.py:39) controls descriptor settings, entropy settings, CPU thread count, and backend routing.

Its `device` option is `auto` by default. `cpu` keeps execution on the CPU and does not initialize torch or CUDA. `gpu` requires a usable CUDA device and torch; `auto` uses GPU when available and otherwise falls back to CPU. GPU support is optional and is not installed by the base requirements. The configuration is stored as split provenance.

## Cross-test assignments

Each split result records `extra_tested_groups`. If the input grouped domain has `specify_cross_tests`, the splitter uses that explicit mapping regardless of `add_extra_cross_tests`, after removing each source group's self-reference and deduplicating its targets. Otherwise, when `add_extra_cross_tests` is `True`, every group is assigned every other group as an extra test group; when it is `False`, no extra cross tests are assigned. The automatically generated other-group target ordering is not guaranteed because it is derived from sets. Extra group datasets remain separate from `SplitGroup.test_set` and are reconstructed for export through `extra_tested_groups`. See [Data format and grouping](raw-data-format.md#implemented-cross-test-behavior) for the metadata details.
