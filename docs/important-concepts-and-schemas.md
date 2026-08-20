# Important concepts and schemas

[Back to the project overview](../README.md) · [Raw data format and grouping](raw-data-format.md) · [Data splitting](data-splitting.md) · [Roadmap](roadmap.md)

TEMPER's implemented workflow turns labeled source structures into reproducible, exported datasets. It deliberately separates **data identity and provenance** from the atomic structures themselves: metadata records describe source files, split records store lightweight frame references, and export materializes the referenced frames into `extxyz` files. This guide explains the objects at each layer and the relationships between them.

## Workflow at a glance

| Stage | Main representation | Contains | Produces / enables |
| --- | --- | --- | --- |
| Source domain | A directory with `metadata.json` and top-level `.extxyz` files | Labeled ASE-readable frames plus human-authored metadata | One [`InfoEntry`](../src/temper/schemas/info.py:20) per source file |
| Grouping | [`GroupedDomain`](../src/temper/schemas/group.py:17) | Domain inventory, grouping strategy, and a mapping from group names to source filenames | One ordered frame pool per group |
| Splitting | [`SplitGroup`](../src/temper/schemas/split.py:299) | Provenance, one train/validation trajectory, own-group test references, and cross-test assignments | Nested checkpoints for one group and repeat |
| Reconstruction / export | [`FrameReferenceResolver`](../src/temper/splitting/io.py:54) and exported `.extxyz` files | Reconstructed labeled frames | One [`TrainingUnit`](../src/temper/schemas/train_unit.py:11) per training checkpoint |

```text
Domain directory
  ├─ metadata.json + source .extxyz files
  └─ InfoEntry (one per file)
          │
          ▼
GroupedDomain (one per configured grouping strategy)
  └─ groups: group name → source filenames
          │
          ▼
SplitGroup (one group × one repeat)
  ├─ test_set: FrameReference[]
  └─ TrainValSplitTrajectory
       └─ nested train checkpoints → train / validation references
          │
          ▼
reference resolution and extxyz export
          │
          ▼
TrainingUnit (one MLFF training attempt)
```

## 1. Domain source data and `InfoEntry`

A **domain** is a single directory below the data root. It holds top-level `.extxyz` source files and a `metadata.json` file. The domain name is part of every later frame identity. The required layout and metadata contract are described in [Raw data format and grouping](raw-data-format.md).

An [`InfoEntry`](../src/temper/schemas/info.py:20) is the inventory record for **one source `.extxyz` file**, not for one individual frame. It combines human-authored provenance with facts detected from the file:

- Required authored identity/context fields are `name`, `source`, `domain`, `filename`, and `system_type`.
- Optional descriptive/provenance fields include the description, first-principles software and settings, theory level, structure-generation method, and additional information. Missing optional values emit warnings.
- [`InfoEntry.from_extxyz`](../src/temper/schemas/info.py:127) derives the initial `name`, `domain`, and `filename` from the source path, and inventories systems, frame counts, atom counts, formulas, stress availability, and other properties. User-provided values can override auto-detected fields, although this is warned against.
- For this inventory, a dpdata-like *system* is inferred by the pair of atom count and Hill-formula composition. It is an inventory concept, distinct from both a source file and a later data group.

[`load_info_entries_from_datadir`](../src/temper/schemas/info.py:218) reads the metadata `info` list, checks that its number of entries equals the number of top-level `.extxyz` files, and uses metadata `filename` values to establish source-file order. It then reconstructs each entry from the corresponding source file. Consequently, `InfoEntry` remains a file-level description backed by the current source tree.

## 2. Grouping: `GroupedDomain` and its constituent representations

Grouping answers: **which source files should be pooled before making a split?** [`partition_domain_into_groups`](../src/temper/grouping/group.py) reads the `groupings` specifications from a domain's metadata and returns one [`GroupedDomain`](../src/temper/schemas/group.py) for each specification. It reuses previously loaded `InfoEntry` objects when processing additional specifications.

A `GroupedDomain` contains these related group-level representations:

| Representation | Level | Meaning |
| --- | --- | --- |
| `info_entries: list[InfoEntry]` | domain / source-file inventory | The full per-file inventory from which groups are formed. |
| `grouping_strategy: str` | grouping definition | The registered strategy used to partition the source filenames. |
| `groups: dict[str, list[str]]` | group | A group name mapped to the `.extxyz` filenames that belong in its split pool. Each filename is validated to have the `.extxyz` suffix. |
| `FrameReference` pool | group / frame | The concrete, ordered, reference-only frame identities derived from every file assigned to a group. |
| `add_extra_cross_tests` and `specify_cross_tests` | group-to-group testing policy | Recorded policy inputs for extra tests; the splitter's implemented behavior is described below. |

The registered filename-based strategies are defined in [`GROUPING_STRATEGIES`](../src/temper/grouping/strategies.py:446): one group per file, explicit groups, all files together, regex/property-based grouping, and NEB generalization grouping. See [Raw data format and grouping](raw-data-format.md) for the metadata syntax and strategy details.

[`GroupedDomain.load_frame_references_in_groups`](../src/temper/schemas/group.py:168) turns each grouped filename into zero-based frame identities. For each source file, it uses the `InfoEntry` inventory's total frame count and creates one [`FrameReference`](../src/temper/schemas/split.py:15) per index. Thus a group starts as file membership but becomes a frame-level pool for splitting. The method's return annotation says it maps group names, but its current implementation populates the result using filenames as keys; the splitter consumes those keys as its group identifiers. This is implemented behavior to be aware of when inspecting results.

### Cross-test assignments

A `SplitGroup` retains its own group's holdout set separately from extra tested groups. In the current splitter, `specify_cross_tests`, when present, takes precedence over `add_extra_cross_tests` and is used as the explicit mapping after each source group's self-reference is removed and its targets are deduplicated. Without that mapping, `add_extra_cross_tests=True` assigns every group every other group as an extra test group, while `add_extra_cross_tests=False` assigns none. The automatically generated target order is not guaranteed because the implementation uses sets. Extra group datasets are added only during export when requested; they are never merged into the split's own `test_set`. See the [implemented cross-test behavior](raw-data-format.md#implemented-cross-test-behavior) for the source-data perspective.

## 3. Splitting: references, trajectories, and repeats

[`split_grouped_domain`](../src/temper/splitting/split.py) performs the split for every group and every configured repeat. [`SplitConfig`](../src/temper/schemas/split.py) supplies the source and output roots, selected domains, repeat count, a reproducible base seed, test ratio, requested training ratios, maximum training size, train/validation method, export options, and [`QuestsAdapterConfig`](../src/temper/schemas/quests_adapter.py). The base seed deterministically derives and persists separate per-repeat seed lists for the two partitioning stages; explicitly loaded lists are reused unchanged.

For each `(group, repeat_id)` pair, the implementation first makes a seeded random train+validation versus test partition. Its own-group test size is `round(pool_size * test_ratio)` using Python's built-in rounding. It computes QUESTS descriptors for the whole group, then selects training frames from the train+validation partition using either `random` or `quests`. The result is one [`SplitGroup`](../src/temper/schemas/split.py:299), not a separate split object per requested training size.

### `FrameReference`: the persistent frame identity

A [`FrameReference`](../src/temper/schemas/split.py:15) contains only:

1. `domain` — a non-empty domain name;
2. `filename` — a safe, relative `.extxyz` path beneath that domain; and
3. `frame_index` — a nonnegative, zero-based index in that source file.

It stores no `ase.Atoms` object and no descriptor. Its `identity` tuple is `(domain, filename, frame_index)`, which schemas use for duplicate and overlap checks. This is the common unit that connects source data, group pools, train/validation selections, and tests.

### `TrainValSplitTrajectory`: one ordered, nested selection

A [`TrainValSplitTrajectory`](../src/temper/schemas/split.py:155) represents the single train/validation selection trajectory within one `SplitGroup`. Both supported methods use the same structure:

- `requested_train_sizes` is a non-empty, strictly increasing list of checkpoint sizes.
- `selected_frames` is ordered. At checkpoint `i`, its prefix of length `requested_train_sizes[i]` is the training set.
- `additional_trainval_frames` contains train+validation frames that were not selected up to the largest requested training size.
- Validation at checkpoint `i` is the remaining `selected_frames` suffix followed by `additional_trainval_frames`.

The checkpoints are therefore **nested**: a smaller training set is an ordered prefix of every later one, and all checkpoints share the same test set. [`get_train_set`](../src/temper/schemas/split.py:256) and [`get_val_set`](../src/temper/schemas/split.py:274) expose these reference lists. A required non-`None` trajectory seed records the selector's randomized initialization; random selection also uses it at later increments.

The optional [`EntropyProfile`](../src/temper/schemas/split.py:112) records ordered QUESTS entropy points for selection steps. A point can cover multiple selected frames, so profile points do not necessarily coincide one-for-one with requested training checkpoints. Cumulative entropy and information gain must be finite, but neither is constrained to be nonnegative or monotonic because of the QUESTS quantities represented.

### `SplitGroup`: provenance for a group-repeat result

[`SplitGroup`](../src/temper/schemas/split.py:299) is the persisted top-level result for one group and one repeat. It includes:

- identity/provenance: `domain`, `grouping_strategy`, `group_name`, and nonnegative `repeat_id`;
- `test_set`: references held out from the current group only;
- `extra_tested_groups`: names of other groups designated for additional testing;
- split configuration evidence: `test_ratio` and `trainval_test_split_seed`;
- exactly one `train_val_split_trajectory`; and
- optional `quests_adapter_config`, which preserves descriptor, entropy, device, and reproducibility settings used for QUESTS work. It can be `None` for older or unevaluated persisted schemas.

A repeat is an independent train+validation/test split of the same group, indexed by `repeat_id`. Repeats are orthogonal to checkpoints: each repeat has one trajectory with many nested checkpoints. The schema checks that all train, validation, and test references have the declared domain, do not duplicate one another, and that the test-set size matches the requested ratio calculation.

Every `SplitGroup` stores a system-managed `split_id`. This deterministic UUIDv5 fingerprints the materialized split definition, including its provenance, frame membership, repeat, and QUESTS configuration, but excluding the derived entropy profile. A serialized ID is verified once when loaded and then reused. Validated reassignment of an identity-defining field regenerates it; tuple-backed reference collections prevent undetected `append`-style mutation and can be replaced as whole fields. Cross-test group order is normalized because it is not semantically significant.

## 4. Reference-based persistence and reconstruction

The principal split schemas derive from [`MSONableModel`](../src/temper/schemas/base.py), which combines Pydantic validation with Monty-compatible dictionaries. Identity-bearing aggregate roots use its `ManagedIdentityModel` child, which centralizes stored-ID verification, assignment regeneration, and validated copying. Persist them with `monty.serialization.dumpfn` and restore them with `loadfn`. Persisted grouping/split information is compact because it records metadata and references rather than embedding source structures or computed descriptors.

### Developing a `ManagedIdentityModel` subclass

A new persisted aggregate with a managed identity should declare the UUID field as a normal Pydantic field and configure the identity lifecycle through class variables:

```python
class ExampleRecord(ManagedIdentityModel):
    _IDENTITY_FIELD_NAME = "example_id"
    _IDENTITY_SOURCE_FIELDS = (
        "domain",
        "nested_record.method",
    )
    _IDENTITY_NAMESPACE = UUID("...")
    _IDENTITY_SCHEMA = "temper.example-record.v1"
    _IDENTITY_LABEL = "example record"

    domain: str
    nested_record: NestedRecord
    example_id: UUID | None = None
```

`_IDENTITY_SOURCE_FIELDS` accepts simple field names and dotted attribute paths. The base class builds the nested canonical payload, derives which top-level assignments invalidate the ID, verifies a loaded ID, computes a missing ID, and serializes the managed UUID as a string. Give each logical schema its own fixed namespace and increment `_IDENTITY_SCHEMA` only when intentionally changing identity semantics. Declare only the required nested trajectory fields rather than their containing model when derived or presentation-only fields must be excluded.

If a source collection has schema-specific equivalence rules, declare the smallest necessary callable in `_IDENTITY_SOURCE_NORMALIZERS`. For example, `SplitGroup.extra_tested_groups` is sorted and deduplicated because cross-test order is not meaningful. Ordered frame collections are not normalized this way because their order is part of their identity.

There is one required validation convention. Do not add an independent `@model_validator(mode="after")` or `mode="wrap"` validator to a `ManagedIdentityModel` subclass, and do not override `_finalize_managed_identity`. Override `_validate_before_identity()` for cross-field or model-wide checks instead, raising `ValueError` for invalid state. The inherited `_finalize_managed_identity` validator calls that hook first and only then verifies or generates the ID. Field validators and `mode="before"` model validators remain appropriate for parsing and individual-field normalization. The base class rejects incompatible validator declarations when a subclass is created, preventing identity from being finalized before subclass consistency checks have completed.

To use a persisted reference later, [`FrameReferenceResolver`](../src/temper/splitting/io.py:54) resolves it as `root_path / domain / filename`. It validates the domain and resolved path boundaries, reads a source `.extxyz` file at most once for the resolver's lifetime, checks frame bounds, and verifies that each loaded frame has energy and forces labels. Reconstruction preserves reference order; cached source frames can be shared, so callers should treat them as read-only. The helpers [`load_frames_from_references`](../src/temper/splitting/io.py:250), [`load_frames_test`](../src/temper/splitting/io.py:302), and [`load_frames_train_validation`](../src/temper/splitting/io.py:346) provide the corresponding operations.

This design makes a split reproducible only relative to the source data tree and its path convention. Keep the source files available under the same domain/relative-filename structure when reconstructing a saved split.

## 5. Export and `TrainingUnit`

[`write_all_sets_in_split_group_to_extxyz`](../src/temper/splitting/io.py:575) materializes every requested training checkpoint from one `SplitGroup`. It writes one training file per checkpoint, optionally writes each non-empty validation file, writes the current group's test file, and can include exported tests from its assigned extra groups. Names are deterministic and writes atomically replace generated artifacts.

Each returned [`TrainingUnit`](../src/temper/schemas/train_unit.py:11) is the exported unit corresponding to **one MLFF training attempt**:

- It identifies one domain, grouping strategy, group, splitting method, repeat, and `n_train` checkpoint.
- It points to exactly one training `.extxyz` file, an optional validation `.extxyz` file, and one or more test `.extxyz` files.
- It records the `split_id` of the `SplitGroup` that produced it and stores its own system-managed, deterministic `training_unit_id`.
- Its fields remain mutable through validated reassignment. Reassigning an identity-defining field regenerates `training_unit_id`; changing only `root_path` does not.
- Its validation ensures that every referenced dataset file exists beneath `root_path / domain` and has the `.extxyz` extension.

All `TrainingUnit` objects made from a given `SplitGroup` share that split's own test file and any selected extra-test files, but differ in their nested training checkpoint and optional validation file. The class describes input datasets for a future training run; it does not create, execute, or evaluate that run.

The `training_unit_id` fingerprints the parent split identity and the unit's dataset contract, but deliberately excludes `root_path`; moving an exported tree therefore does not rename its logical units. The stored value is verified once during loading and reused afterward. Older manifests without IDs remain loadable and receive stable identities from their existing fields, but only newly exported units can distinguish identical coordinates produced by different split definitions.

## Implemented scope versus roadmap

The implemented Python API and split CLI cover metadata loading, file grouping, repeatable reference-based splitting, QUESTS-backed selection and provenance capture, reconstruction, and `extxyz` export. Training-job creation, MLFF training orchestration, benchmark evaluation, and metrics calculation are **not implemented**. They are explicitly planned capabilities in the [Roadmap](roadmap.md). Therefore, `TrainingUnit` should be read as an exported training-data contract, not as evidence that TEMPER currently schedules or performs MLFF training.

## Further reading

- [Project overview](../README.md) — implemented workflow and scope.
- [Raw data format and grouping](raw-data-format.md) — domain layout, `metadata.json`, `InfoEntry`, grouping strategies, and cross-test configuration.
- [Data splitting](data-splitting.md) — public Python workflow, configuration, reconstruction, export, and QUESTS backend details.
- [Roadmap](roadmap.md) — capabilities that remain planned rather than implemented.
