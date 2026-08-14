# Data splitting

[Back to the project overview](../README.md) · [Data format and grouping](data-format.md) · [Roadmap](roadmap.md)

The `temper` splitter turns a grouped data group into train, validation, and test sets. Split results are persisted as reference-only `SplitDataSchema` objects: they store only `(domain, relative extxyz source filename, nonnegative frame index)`, never structures or descriptors. There is no persisted `schema_version` or top-level `trainval_pool` field.

See [`src/temper/schemas/split.py`](../src/temper/schemas/split.py) for the schema definitions.

## Schemas and inventory

- `FrameReference` is one persisted reference to a structure frame.
- `TrainValSplitTrajectory` is the ordered trajectory for one method, either `random` or `quests`. Its complete train/validation inventory consists of ordered `selected_frames` followed by ordered `additional_trainval_frames`. Prefixes of `selected_frames` define nested training sets at `requested_train_sizes`. A trajectory may also contain an `EntropyProfile`.
- `SplitDataSchema` is the persisted result for one splitting method. It has exactly one singular `train_val_split_trajectory`, plus the reference-only `test_set` and split provenance.
- `EntropyProfile` and `EntropyProfilePoint` contain QUESTS maximum-entropy evaluation data: cumulative entropy and information gain at each requested size.
- `QuestsSplitConfig` contains typed QUESTS descriptor, entropy, and device configuration and is stored as schema provenance in `quests_config`.
- `SplitSchema` is the legacy configuration-oriented schema retained for backward compatibility.

## High-level API

The entry point is `split_data_group` in [`src/temper/experiments.py`](../src/temper/experiments.py). It:

1. accepts a group's per-file frame indices and aligned `ase.Atoms` structures;
2. verifies structure ordering against generated `FrameReference` objects;
3. performs the initial train+validation versus test partition, which is always random;
4. normalizes requested training sizes; and
5. evaluates each selected splitting method.

It **always returns a list** containing one `SplitDataSchema` per requested method, including a one-element list when only one method is requested.

```python
from src.temper.experiments import split_data_group
from src.temper.schemas import QuestsSplitConfig

schemas = split_data_group(
    frames_by_filename={"a.extxyz": [0, 1, 2], "b.extxyz": [0, 1, 2]},
    structures_by_filename={"a.extxyz": atoms_a, "b.extxyz": atoms_b},
    domain="sse_llzo",
    grouping_strategy="all",
    group_name="all",
    split_seed=7,
    train_val_split_method=["random", "quests"],
    quests_config=QuestsSplitConfig(),
    test_ratio=0.2,
    requested_train_sizes=[0.25, 0.5, 1.0],  # ratios of the train+validation pool
    random_seed=3,
)
random_schema, quests_schema = schemas
```

### Method selection and sizing

- `train_val_split_method` is an explicit sequence containing `random`, `quests`, or both, for example `['random', 'quests']`. The result list follows this order. Each result has one trajectory, not a collection of trajectories.
- The test set is controlled by `test_ratio`, whose default is `DEFAULT_TEST_RATIO`, or by `test_size`.
- `requested_train_sizes` values are ratios of the train+validation pool when `as_ratio=True`, the default, or exact integer counts when `as_ratio=False`.
- `max_train_size`, whose default is `DEFAULT_MAX_N_TRAIN`, caps training sizes. When necessary, requested ratios are scaled down proportionally. Defaults are defined in [`src/temper/utils/env.py`](../src/temper/utils/env.py).
- `random_seed` is required when `random` is selected. The train+validation versus test partition always uses `split_seed`.

## Train and validation semantics

1. The train+validation versus test partition is always random and uses `split_seed`.
2. The complete train/validation inventory is ordered as `selected_frames + additional_trainval_frames`.
3. At checkpoint index `i`, the training set is the prefix of `selected_frames` whose length is `requested_train_sizes[i]`.
4. Validation is the remaining selected suffix followed by every additional frame: `selected_frames[requested_train_sizes[i]:] + additional_trainval_frames`. This preserves inventory order and makes additional frames available at every checkpoint.
5. `random` and `quests` use the same train+validation pool and requested sizes. The random trajectory's entropy profile is evaluated with the same QUESTS objective used for QUESTS selection.

`TrainValSplitTrajectory.get_train_set(i)` and `TrainValSplitTrajectory.get_val_set(i)` take a **checkpoint index** into `requested_train_sizes`. The IO APIs use this same index directly; they neither dispatch by method nor accept a training-size value.

## QUESTS backend

QUESTS is substantially faster on a GPU, so a CUDA-capable GPU is recommended. The maximum-information-entropy method uses `quests==2026.2.22`. Torch and CUDA support are optional rather than requirements for a CPU-only installation.

`QuestsSplitConfig.device` defaults to `auto`:

- `auto` chooses the GPU route when torch is installed and CUDA is usable, otherwise it safely falls back to CPU.
- `cpu` forces CPU operation without importing or initializing CUDA or torch.
- `gpu` requires an available CUDA device and fails when one is unavailable. This route requires the optional `quests[gpu]` extra, or an equivalent torch installation, and usable CUDA.

Backend modules are loaded lazily. CPU splitting does not import torch, and GPU modules are imported only when the GPU route is selected. A persisted `device='auto'` value records a policy, not the backend that created the split; the actual backend depends on torch and CUDA availability at runtime.

Other `QuestsSplitConfig` fields control:

- descriptor parameters: `descriptor_k`, `descriptor_cutoff`, and `descriptor_dtype`;
- entropy parameters: `entropy_bandwidth`, `entropy_batch_size`, and `entropy_tolerance`; and
- optional process-wide `numba_threads` control for CPU kernels.

## Persistence and reconstruction

Persistence stores only frame references; source structures and descriptors are never embedded. A `SplitDataSchema` contains one singular `train_val_split_trajectory`, not a top-level train/validation pool or list of trajectories.

[`src/temper/splitting/io.py`](../src/temper/splitting/io.py) reconstructs references into ordered, labeled `ase.Atoms` frames from `root_path / domain / filename`:

- `SourceResolver(root_path=DEFAULT_DATA_DIR)` binds the source tree and caches source files.
- `load_frames_from_references(references, root_path)` reconstructs ordered labeled frames.
- `load_frames_test(schema, root_path)` reconstructs the test set.
- `load_frames_train_validation(schema, requested_size_index, root_path)` returns `(train, validation)` for a checkpoint index into `requested_train_sizes`.

## Export

- `build_export_filename(...)` creates deterministic names of the form `<domain>__<strategy>__<group>__<method>__<role>__n<count>.extxyz`.
- `write_single_dataset_to_extxyz(...)` writes one non-empty labeled set.
- `write_all_sets_in_split_schema_to_extxyz(schema, output_dir, root_path, *, write_validation=False)` writes every training checkpoint and the test set.

The `write_validation` argument is keyword-only and defaults to `False`. Validation export is therefore opt-in: pass `write_validation=True` to write validation checkpoints, and empty validation sets are skipped rather than written. On each call, generated files atomically replace prior artifacts.

## Reproducibility

A split is fully reproducible given the same source inventory order, split and random seeds, requested training sizes, and `QuestsSplitConfig`. QUESTS selection is deterministic: it is greedy, resolves ties by pool order, and stores no seed.

## Related documentation

The source inventory and grouping metadata consumed by this workflow are described in [Data format and grouping](data-format.md). Planned project-level capabilities are listed in the [Roadmap](roadmap.md).
