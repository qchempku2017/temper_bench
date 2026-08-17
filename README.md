# TEMPER benchmark

TEMPER provides the implemented data-preparation layer for a benchmark of machine-learned force fields (MLFFs). It reads labeled `extxyz` data, organizes each domain into groups, creates reproducible train/validation/test splits, and exports referenced frames back to `extxyz` files.

## What is implemented

The current public workflow is Python API based:

1. Prepare a domain directory containing labeled `extxyz` files and a `metadata.json` file.
2. Call [`partition_domain_groups`](src/temper/grouping/group.py:15) to load the metadata and produce one grouped domain per configured grouping strategy.
3. Call [`split_grouped_domain`](src/temper/splitting/split.py:237) with a [`SplitConfig`](src/temper/splitting/split.py:128) to split every group for every configured repeat.
4. Persist split models with their JSON helpers, or reconstruct and export datasets with [`FrameReferenceResolver`](src/temper/splitting/io.py:87) and [`write_all_sets_in_split_group_to_extxyz`](src/temper/splitting/io.py:608).

Split records store frame references rather than embedded structures or descriptors. A [`FrameReference`](src/temper/schemas/split.py:34) identifies one source frame by domain, relative `extxyz` filename, and zero-based frame index. This keeps the split compact and lets it be reconstructed from the original data tree.

## Current scope

Grouping, splitting, QUESTS-backed selection, split persistence, frame reconstruction, and `extxyz` export are implemented. The split entry point in [`src/temper/entrypoints/split.py`](src/temper/entrypoints/split.py) is only a TODO stub: there is no command-line interface. Training jobs, benchmark execution, result uploading, and metrics are not implemented features.

## Documentation

- [Data format and grouping](docs/raw-data-format.md) — required domain layout and metadata, inventory autodetection, grouping strategies, and cross-test behavior.
- [Data splitting](docs/data-splitting.md) — split configuration, result schemas, Python API, QUESTS configuration, reference-based reconstruction, and export.
- [Roadmap](docs/roadmap.md) — clearly marked planned capabilities.
