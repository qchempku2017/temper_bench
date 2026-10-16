# Roadmap

[Back to the project overview](../README.md) · [Data format and grouping](raw-data-format.md) · [Data splitting](data-splitting.md) · [Local MLFF bundles](mlff-bundles.md)

The following capabilities are planned and are **not** currently implemented:

- benchmark metric calculation from standardized raw predictions;
- a web platform for uploading custom datasets;
- a web UI for authoring dataset information and grouping metadata, including inventory-field autodetection from uploaded `extxyz` files;
- an online repository for split definitions and datasets;
- result uploading; and
- benchmark-result visualization.

The implemented Python workflow and split CLI cover data metadata, grouping,
split generation, reference-based reconstruction, and content-hashed `extxyz`
exports. The MLFF layer defines `MLFFSpec`, creates `MLFFTrainBundle` records,
and writes fine-tuning or zero-shot folders for DPA-4, DPA-4C, MatterSim, MACE,
SevenNet, and NEP-89. The installed runtime generates native training files and
executes training and ASE evaluation. [Remote jobs](remote-jobs.md) adds Slurm,
PBS and Bohrium submission, JSON attempt history, optional watching, recovery,
explicit bulk retries and verified result collection followed by remote cleanup.
Resource-specific live smoke tests remain necessary before production use.
