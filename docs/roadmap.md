# Roadmap

[Back to the project overview](../README.md) · [Data format and grouping](raw-data-format.md) · [Data splitting](data-splitting.md)

The following capabilities are planned and are **not** currently implemented:

- a command-line interface for the split workflow;
- training-job creation and MLFF training orchestration;
- benchmark evaluation and metrics calculation;
- a web platform for uploading custom datasets;
- a web UI for authoring dataset information and grouping metadata, including inventory-field autodetection from uploaded `extxyz` files;
- an online repository for split definitions and datasets;
- result uploading; and
- benchmark-result visualization.

The implemented Python workflow currently covers data metadata, grouping, split generation, reference-based persistence and reconstruction, and `extxyz` export. See [Data format and grouping](raw-data-format.md) and [Data splitting](data-splitting.md) for the available behavior.
