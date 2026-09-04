# Roadmap

[Back to the project overview](../README.md) · [Data format and grouping](raw-data-format.md) · [Data splitting](data-splitting.md) · [Local MLFF bundles](mlff-bundles.md)

The following capabilities are planned and are **not** currently implemented:

- remote upload through SSH/SCP/rsync or provider APIs;
- remote-host, hardware, and scheduler configuration;
- Slurm/PBS submission, job identifiers, polling, and heartbeat monitoring;
- `MLFFTrainRun` execution records, retries, and resubmission;
- result downloading and local result organization;
- benchmark metric calculation from standardized raw predictions;
- a web platform for uploading custom datasets;
- a web UI for authoring dataset information and grouping metadata, including inventory-field autodetection from uploaded `extxyz` files;
- an online repository for split definitions and datasets;
- result uploading; and
- benchmark-result visualization.

The implemented Python workflow and split CLI cover data metadata, grouping, split generation, reference-based persistence and reconstruction, and `extxyz` export. The local MLFF layer also defines `MLFFSpec`, creates atomic `MLFFTrainBundle` records, and writes self-contained fine-tuning or zero-shot folders for DPA-4, DPA-4C, MatterSim, MACE, SevenNet, and NEP-89. Those folders include native training instructions and a standardized ASE prediction runtime, but TEMPER does not upload or execute them. See [Local MLFF bundles](mlff-bundles.md) for the exact boundary.
