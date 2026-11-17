# Roadmap

[Back to the project overview](../README.md) · [Data format and grouping](raw-data-format.md) · [Data splitting](data-splitting.md) · [Local MLFF bundles](mlff-bundles.md)

The following capabilities are planned and are **not** currently implemented:

- result metrics, experiment comparisons and learning curves;
- data-efficiency scores, such as the labeled-data budget needed to reach a
  target error, with explicit interpolation and missing-data rules;
- aggregation and uncertainty estimates across repeated learning curves;
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

[Result loading](results-loading.md) now preserves labeled test datasets
before managed-job cleanup. The runtime writes one root `results.json` with
training state and time plus named `test_manifests`; managed collection downloads
it and preserves partial results after failures. `MLFFExperimentManifest` and its
test entries are `MSONableModel` schemas created by the pipeline runner and
passed explicitly to evaluation. The shared parent methods `from_json_file()`
and `to_json_file()` read and write individual models through `from_dict()` and
`to_dict()`. Result manifests, bundles and split configurations all use the
shared atomic writer in `temper.utils.file_writing`. Training status and duration
live directly on `MLFFExperimentManifest`. It references the settings file rather than
embedding the bundle.
Manifest reads and local loading do not track or check the manifest's own checksum.
`load_results(base_dir=...)` takes the result directory explicitly and returns a separate
`MLFFExperimentResult` with settings, measurements, checked frames, predictions
and reference labels. Neither `MLFFExperimentManifest` nor its `MLFFTestManifest` recipes
contain scientific arrays. Execution and collection do not compose loaded data.
Prediction NPZ files contain numerical arrays only. Test descriptions, properties,
units and evaluation durations live in `MLFFTestManifest`; package versions are
recorded once in `MLFFExperimentManifest`. Frame and atom counts are read from
the arrays rather than stored again as metadata.
Portable export is implemented in `temper.results_loading`, alongside the public
`result_loading` module used by the manifest methods. These capabilities
support later analysis. Each result keeps the original exported test filename
for both own-group and cross-group tests. Callers retain split-group records;
results do not duplicate their group definitions or original frame references.
Successful fine-tuning records only the elapsed training-call time; missing
or failed measurements remain unavailable rather than becoming zero.

Successful calculations, tests, training and cleanup use `completed` throughout
the runtime and job records. Submission decisions use `operation_failed` for
operational errors, separately from a calculation's `failed` state.

All persisted schema versions and identity-schema definitions start at 1.
Only the current format is supported; no compatibility import or migration
code is maintained.

Local result loading does not require an installed MLFF backend or a remote
connection. Result-analysis methods remain future work.
