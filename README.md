# TEMPER benchmark

TEMPER is a systematic benchmark of the fine-tuning capabilities of machine-learned force-field architectures (MLFFs).

## Benchmark workflow

A complete benchmark has three stages:

1. **Data splitting** — generate shared training, validation, and test datasets that are reused for every MLFF.
2. **Training and validation** — compose Bohrium jobs that train MLFFs on the training datasets and validate them on the test datasets.
3. **Metrics** — calculate evaluation metrics from the test results.

## Major concepts

Input data is organized into domain directories containing labeled `extxyz` files and a `metadata.json` inventory. Metadata records dataset provenance and defines grouping strategies, including optional cross-group tests.

For each grouped data group, TEMPER first creates a shared random train+validation versus test partition. It then produces nested training checkpoints with random selection, QUESTS maximum-information-entropy selection, or both. Persisted split schemas contain frame references and provenance rather than embedded structures or descriptors, enabling deterministic reconstruction and export from the original source tree.

## Saving and loading schemas

All schemas in src/temper/schemas supports saving and loading to/from JSON files via `save_json` and `load_json` methods.`

## Documentation

- [Data format and grouping](docs/data-format.md) — domain layout, `extxyz` requirements, metadata fields, dpdata conventions, grouping strategies, and cross-test behavior.
- [Data splitting](docs/data-splitting.md) — schemas, API, train/validation semantics, QUESTS configuration, persistence, reconstruction, export, and reproducibility.
- [Roadmap](docs/roadmap.md) — planned upload, repository, UI, and visualization capabilities.
