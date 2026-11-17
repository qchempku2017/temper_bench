# Recording and loading results

[Project overview](../README.md) · [MLFF bundles](mlff-bundles.md) · [Remote jobs](remote-jobs.md)

TEMPER retains predictions and experiment settings and reads reference labels
from the user's original split datasets. It provides result recording and loading; metric calculation,
experiment comparison and learning curves are not implemented. Reading results requires the base TEMPER
dependencies, including NumPy and ASE. It does not require CUDA, an MLFF
package, QUESTS, or a connection to the execution machine.

## What is retained

**Keep the original split datasets unchanged for as long as you need to analyze
results. This is the user's responsibility.** TEMPER does not create a reference
archive or another permanent copy of these datasets. Missing or modified files
cause scientific result loading to fail.

Packaging copies test files into `testing_inputs/`, preserving their original
filenames and any relative subdirectories. The remote runner reads these temporary
copies. Local copies can be removed after remote upload, and execution copies are
removed during cleanup. Predictions use the same filenames with a `.npz` extension
under `testing_outputs/`.

For local analysis, the original reference file is resolved as
`bundle.training_unit.root_path / test.source_domain / test.source_filename`.
Packaging records the absolute original split-data root, so changing the working
directory or moving the result folder does not change the reference location.
The loader checks the file's recorded SHA-256 before reading its labels.

Each calculation writes one root `results.json` containing a serialized
`MLFFExperimentManifest`, a Pydantic model derived from `MSONableModel`.
The runner creates it before preparation, saves completed training time before
evaluation, and updates it after each finished test. It marks a caught failure
as `failed` while keeping earlier completed tests. An abrupt termination can
leave a `running` record; this does not establish that the process is still
running. Prediction values remain in separate NPZ files.

`temper.mlff.pipeline.runner.make_test_manifests(bundle, ...)` supplies the pending test entries.
Each entry already contains `execution_input_path` and the reference checksum, requested
properties, units and planned prediction path. Its prediction checksum and
duration are initially `None`. `run_test(manifest, bundle_root=...)` reads these
instructions directly, using the referenced bundle for shared calculator and
model settings. No intermediate `test_config.json` is generated or read.

Successful experiment, training and test states all use `completed`. A loaded
`MLFFExperimentResult` uses the same execution and training status types as its
manifest.

New runs do not create separate dataset-metadata JSON, training-timing JSON
or test-summary JSON files. The manifest stores training state and duration,
a path and checksum for the experiment settings, and `test_manifests`: one
`MLFFTestManifest` per named dataset. Each test entry records its state, original
name, temporary input and prediction locations, checksums, requested properties,
units and evaluation duration. Package versions are recorded once in the
experiment manifest. Calculator and model settings remain in the referenced
bundle. NPZ files contain only energies, forces, optional stresses, atom offsets
and frame indices. Frame and atom counts are obtained from these arrays.
Loading never requires pickle.

Managed collection downloads `results.json` and its referenced `bundle.json`.
The runner's manifest is required even for failed or cancelled runs. If it was
not downloaded, collection records an error and leaves remote files and logs
in place; it never creates a replacement manifest. Collection checks the
downloaded bytes and verifies the settings and predictions before marking
collection verified. It keeps `results.json` and `bundle.json` unchanged, including
their filenames. It neither rewrites paths nor reconstructs measurements.
The same verification runs with `download_only=True`, without remote cleanup.
Collection does not read or copy original reference datasets; they are used only
when loading scientific data for analysis.

Prediction paths are relative to `results.json`. Keep `bundle.json`, `results.json`
and the prediction files together when moving results. For managed jobs, keep the
complete job store together so its separate job-history references remain valid.
The original split datasets must remain at the location recorded in the bundle.

### Original filenames and chemical-space groups

Original filenames are used throughout packaging, evaluation and analysis.
Every result entry keeps:

- `source_filename`: the exact exported test filename, including any relative
  path within its original domain; and
- `source_domain`: the original domain.

Keep the corresponding `SplitGroup` records yourself when you need group
definitions or original frame references. Results do not copy that information
or reconstruct missing split groups. The caller is responsible for retaining
those records and relating them to the preserved test filenames.

Named entries remain separate even if two source files have identical bytes.
`dataset_id` equals the original filename, such as `test_own.extxyz`. Use that
filename to select a dataset when loading results. Errors from different test
datasets are never combined automatically.

### Training time

`MLFFExperimentManifest.training_wall_time_seconds` measures only the elapsed time
inside `adapter.train()`. It includes loading, validation and saving performed
inside that call. It excludes separate preparation, evaluation, scheduling
and file transfer. The duration is saved immediately after successful training
and remains available if evaluation later fails.

A failed or interrupted training call has no completed measurement. Its value
is `None`, never zero. `training_status` is `pending`, `running`,
`completed`, or `failed`; zero-shot evaluation uses `not_applicable`.
The default `unavailable` means no training measurement was supplied. Timing belongs to
the execution attempt and does not change the bundle identity.

## Inspect metadata and load results

`MLFFExperimentManifest`, `MLFFTestManifest`, `MLFFExperimentResult` and `MLFFTestResult`
are defined together in `temper.schemas.mlff_experiment_result`. They are also
available directly from `temper.schemas`.

```python
from pathlib import Path
from temper.schemas import MLFFExperimentManifest

path = Path("bundles/mace/domain/bundle__.../attempt_.../results.json")
manifest = MLFFExperimentManifest.from_json_file(path)
print(manifest.training_status, manifest.training_wall_time_seconds)
for test_manifest in manifest.test_manifests:
    print(test_manifest.dataset_id, test_manifest.source_filename)

experiment = manifest.load_results(base_dir=path.parent)  # Load completed tests.
print(experiment.training_wall_time_seconds)
for result in experiment.test_results:
    print(result.manifest.source_filename)
    print(result.predictions["energies"], result.references["energies"])

```

`from_json_file()` reads the saved field dictionary through `from_dict()` without
opening reference datasets, training inputs, model weights or prediction
arrays. `to_json_file(path)` validates the fields, converts them with `to_dict()`,
and saves them using `temper.utils.file_writing.write_json_atomically`.
The runner uses this same API to persist each state change.
Both file methods are inherited from `MSONableModel` and are shared by the
other model classes, including `MLFFTrainBundle` and `SplitConfig`.
Inherited `to_dict()`, `from_dict()`, `as_dict()` and `to_json()` are also available.
Neither the manifest nor its test entries contain loaded bundles or arrays.

The JSON file contains field values without Monty class tags. To use Monty's
generic reader, construct the model from the returned dictionary:

```python
from monty.serialization import loadfn

manifest = MLFFExperimentManifest.from_dict(loadfn(path))
experiment = manifest.load_results(base_dir=path.parent)
manifest.to_json_file(path.with_name("results_copy.json"))
```

`base_dir` is always required. Pass the directory containing `results.json`;
none of the JSON reading or writing methods remembers the file location.

`load_results()` returns a separate `MLFFExperimentResult` with experiment
settings, training measurements and all completed tests in manifest order.
Pending and failed tests retain their status in the manifest and have no loaded
result. Each completed test is
an `MLFFTestResult` containing ASE structures and aligned numerical arrays.
Its `manifest` holds the corresponding `MLFFTestManifest`, including units and
evaluation duration. Shared versions are available as `experiment.package_versions`. The
runner and collection code use manifests, but do not construct loaded experiment
objects; composition happens locally after retrieval.

Loading leaves the manifest and saved files unchanged. Repeated calls create
independent data objects; manifests do not cache scientific data.
Recorded measurements are available directly on the manifest. To read experiment
settings without loading scientific data, use
`load_bundle_from_manifest(manifest, path.parent)` from
`temper.results_loading.manifest_io`. Loading uses the fields of the manifest object without rereading
or checksumming `results.json`. Call `from_json_file()` again to read updated fields
from disk. After moving results, pass their new manifest directory as `base_dir`.

Loading checks checksums, original filenames, recorded identities, frame order,
atom counts, shapes, requested properties and units. Settings and prediction paths
are relative to `base_dir`; references use the original split-data root. Relative
`base_dir` values are resolved against the working directory at loading time.
Pass an absolute path if the working directory may change.

`predictions` and `references` contain `energies`, `forces` and optional
`stresses`. Force boundaries are in `atom_offsets`; `frame_indices`
indexes the exported test file. These fields exist only on loaded test data.

For a failed or unfinished calculation, loading includes every completed test.
The manifest still describes all named tests, including pending and failed
tests whose results are unavailable.

NaN and infinite values raise an error by default. Use
`manifest.load_results(base_dir=path.parent, require_finite=False)` only to inspect such output.
Unchanged file hashes
establish that saved bytes have not changed. They do not by themselves
establish correct physical labels or predictions.

### Metadata helper functions

For metadata access,
`temper.results_loading.manifest_io.load_bundle_from_manifest(manifest, root)`
reads the referenced bundle and checks its checksum, identities and consistency
with the manifest.

For path resolution, `temper.utils.file_paths.resolve_path_in_bundle(root, name)`
resolves a relative path and checks that it stays inside the bundle directory;
the target need not exist.

These functions take `pathlib.Path` directory or file arguments. Relative paths
use the working directory. They do not read original datasets, model weights or
prediction arrays, and do not write files.

Read a manifest with `MLFFExperimentManifest.from_json_file(path)`. When its
bundle is also needed, pass that manifest and `path.parent` to
`load_bundle_from_manifest`.

## Format and failure cases

The result manifest, bundle and job
record all use schema version 1. Only this format is supported. Unknown
versions and removed fields are rejected; there is no automatic migration.

Missing original labels cannot be reconstructed from predictions. Changed
reference files, predictions or experiment settings are rejected by checksum
verification. Missing measurements remain absent rather than being inferred.

There is no archival or export layer in `results_loading`. The public
`temper.results_loading.result_loading` module provides
`load_results_from_manifest(manifest, base_dir=...)`, which assembles local
data for `MLFFExperimentManifest.load_results()`. Reading a manifest or loading its data never
launches a calculation.
