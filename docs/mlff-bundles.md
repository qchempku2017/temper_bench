# MLFF specifications and submit bundles

[Back to the project overview](../README.md) · [Default variables](default_variables.md) · [Important concepts and schemas](important-concepts-and-schemas.md)

TEMPER supports DPA-4, DPA-4C, MatterSim, MACE, SevenNet, and NEP-89.
Local builders create model bundles; `MLFFBundleWriter` copies their inputs and
serializes `MLFFTrainBundle` as `bundle.json`. An installed TEMPER runtime reads
that record, writes missing configs and scripts, prepares backend-dependent
data and checkpoint settings, fine-tunes, and evaluates.
The [remote job commands](remote-jobs.md) handle upload, scheduling, persistent
monitoring, explicit retries, result download and remote cleanup through dpdispatcher.

Install `temper-bench[preprocess]` for local data splitting. Execution hosts
need the base `temper-bench` package plus their selected MLFF backend, using
preferably the same TEMPER release as the writer. Other releases warn on load.
Bash is required for native training.

## Creating bundles from split results

After `temper_bench split`, use `create_bundles` to pair training units with
pretrained models. Each requested model is combined with every selected unit:

```console
temper_bench create_bundles ./split_results --model mace mpa-0-medium --model dpa4 nano-omat24-v20260805 --bundle-write-root ./bundles
temper_bench job submit --bundle-write-root ./bundles --resource-file cluster.yaml
```

Each `--model` takes a family key followed by a variant key from
`default_pretrained_models.json`. Repeat the option for additional models.
The catalog lists weights; it does not download them. Place them beneath
`--pretrained-model-dir` using their catalog filenames before creating bundles.

The split-results argument defaults to `DEFAULT_SPLIT_RESULTS_DIR`
(`./split_results`). The pretrained directory defaults to
`DEFAULT_MLFF_PRETRAINED_MODELS_DIR` (`./pretrained_models`), and the output
root defaults to `DEFAULT_BUNDLE_WRITE_ROOT` (`./bundles`). These environment
variables are read when TEMPER starts. Relative command-line paths use the
current working directory; a leading `~` expands to the user's home directory.

By default, the command reads every immediate domain directory containing
`DEFAULT_TRAINING_UNITS_FILE` (normally `training_units.json`). It uses the
supplied split-results root to locate datasets, even if the directory was moved
after splitting. Written bundles contain absolute source paths, so they can be
submitted from another working directory. Dataset hashes are checked; changed
files are an error, not silently assigned new identities. Older units without
hashes must be regenerated before this command can use them.

Two options narrow the selection and may be used together:

```console
temper_bench create_bundles --model mace mpa-0-medium --domains water silicon --max-train-n-atoms 100000
```

`--domains` selects exact domain directory names. `--max-train-n-atoms` sets an
inclusive limit on the **total atoms across all training frames**, not atoms per
frame. Zero-shot units from selected domains are always included. A limit of
zero therefore selects only zero-shot units. Negative limits, missing requested
domains, and selections containing no units are errors. There is no option for
selecting individual training-unit files or IDs.

Bundles are written using the same directory naming function as job submission:

```text
bundles/
  <model>__<version>/
    <domain>/
      bundle__...__<bundle-id>/
        bundle.json
```

At this stage, `bundle.json` describes an experiment and has `files: null`.
No datasets or weights are copied. `job submit` discovers every bundle-level
bundle under the root and checks its saved job history before acting. It creates
`attempt_001` for new work, reconnects to interrupted or running attempts, and
creates a later attempt only for an eligible retry. Repeating `submit` does not
restart a successful calculation or launch another copy of a running one.
`submit --preview` reports the proposed actions without launching anything.
Explicitly abandoned attempts are excluded from these active checks; abandoning
does not stop a remote job, so investigate it before allowing a replacement.
See [abandoning unresolved attempts](remote-jobs.md#abandoning-an-unresolved-attempt).
It does not treat packaged bundles inside attempts as new inputs.
Each attempt retains its packaged `bundle.json` beside `results.json` for later
analysis. The first submission also saves its packaged file locations in the shared
parent bundle's `files` field for later status checks and result collection.
Later attempts must use that same layout. Neither the bundle nor its file
locations are repeated in `job.json`.
Each job record references that shared bundle by relative path instead of
embedding its full contents. Keep the parent bundle with its attempts and do
not change its scientific identity. `refresh_bundle_identity` rejects bundles
with existing attempts; create a separate bundle for a changed experiment.
Repeated unit/model pairs are deduplicated by bundle ID. Repeating creation
skips IDs already present, leaving their bundles and attempts unchanged. To
use a new dataset location for an existing ID, use a fresh output root or
explicitly update the existing bundle's source paths.

All selected inputs are validated before bundles are written. A later filesystem
write failure can leave some completed bundles; rerunning skips their IDs.
The completion message reports how many bundles were created and skipped.

### Model parameter overrides

Use `--model_override_config overrides.yaml` (JSON is also accepted) to supply
the existing spec builders' `training_parameters` and `testing_parameters`:

```yaml
mace:
  mpa-0-medium:
    training_parameters:
      max_num_epochs: 50
      batch_size: 8
    testing_parameters:
      default_dtype: float64
```

The outer keys are family and variant keys, matching `--model` selections.
Unselected models and unknown setting names at this level are errors. Both
parameter entries must be dictionaries; either may be omitted. Fine-tuning
units use the builder's training defaults with these overrides. Zero-shot units
always disable training and use only testing overrides. Testing overrides apply
to both modes. Existing builder rules still apply, including model-specific
testing requirements and TEMPER-managed training settings. Training dictionaries
are merged at the top level by the builder; a supplied nested value replaces
the corresponding default value as a whole.

## Building a specification

Every builder accepts either a catalog `variant` or all three explicit fields:
`model_name`, `model_version`, and `model_filename`. These fields belong to the
builder instance, so different sizes and pretraining branches can coexist.
There is no implicit default model, and combining a variant with explicit model
fields raises `ValueError`.

Omitting the directory uses `DEFAULT_MLFF_PRETRAINED_MODELS_DIR`, whose default
is `./pretrained_models`. A relative directory is relative to the current working
directory at `build()` time, not the repository or installed package. `~` is
expanded. Checkpoint filenames use relative POSIX paths, including subdirectories.

~~~python
from temper.mlff.spec_builders import MACESpecBuilder
from temper.schemas.mlff_train_bundle import MLFFTrainBundle

spec = MACESpecBuilder(
    variant="mpa-0-medium",
    # Looks for /shared/models/mace/mace_mpa_0/mace-mpa-0-medium.model.
    pretrained_model_dir="/shared/models",
    # None means test the pretrained model without training.
    # An empty dict enables the pinned fine-tuning defaults.
    training_parameters={},
    # Flat native MACECalculator keyword arguments.
    testing_parameters={"default_dtype": "float32"},
).build()

bundle = MLFFTrainBundle(training_unit=training_unit, mlff_spec=spec)
submit_directory, packaged_bundle = bundle.write_submit_folder("submit/mace-run")
~~~

For a custom or renamed local checkpoint, supply its identity explicitly:

~~~python
custom = MACESpecBuilder(
    model_name="my-mace-small-materials",
    model_version="experiment-2026-10-14",
    model_filename="custom/checkpoint.model",
    pretrained_model_dir="/shared/models",
)
~~~

The packaged [default_pretrained_models.json](../src/temper/mlff/default_pretrained_models.json)
contains release metadata only. Download weights separately from each entry's
`download_url` and place them at `pretrained_model_dir / model_filename`.
No checkpoint is downloaded or read during initialization; `build()` reads and
hashes the local file. DeepMD fine-tuning additionally requires the matching explicit JSON beside
the checkpoint, with the same filename stem, unless `training_parameters.model`
is supplied explicitly. Other families do not require this JSON.

~~~python

from temper.mlff.spec_builders.base import pretrained_model_variants, pretrained_model_defaults
from temper.mlff.spec_builders import mlff_spec_builder_factory

variants = MACESpecBuilder.available_variants()  # tuple of case-sensitive keys
metadata = pretrained_model_defaults("mace", "mpa-0-medium")
family_catalog = pretrained_model_variants("mattersim")  # independent dictionaries
builders = [
    mlff_spec_builder_factory("mattersim")(variant=variant)
    for variant in family_catalog
]
~~~

The factory still returns a builder class. Catalog lookup works from installed
distributions and does not depend on the source checkout. A catalog selection resolves to
ordinary spec fields; the runtime does not consult the catalog. Moving identical
checkpoint bytes does not change the specification identity.

### Pretrained release survey (updated 2026-10-27)

The catalog contains 36 checkpoint choices. Every entry has an
integer `estimated_num_parameters`, describing the full pretrained model before
fine-tuning or element/head selection. It is a parameter count, not a byte size.
`parameter_count_notes` explains whether the value is published, nominal, or
estimated from checkpoint tensor shapes. Estimates exclude optimizer state;
SevenNet-omat's rounded estimate includes small model buffers.

| Family | Catalog choices | Parameter range | Sources |
| --- | --- | --- | --- |
| DPA4 | OMat24 Nano/Mini/Neo/Air/Plus (`v20260805`); MatPES R2SCAN Mini/Neo/Air (`v20260628`) | 0.48M–8.85M | [AIS Square OMat24](https://www.aissquare.com/models/detail?pageType=models&id=423&name=DPA4-OMat24) |
| DPA4C | OMat24 Air (`v20260819`) | 433,673 | [AIS Square OMat24](https://www.aissquare.com/models/detail?pageType=models&id=433&name=DPA4C-OMat24) |
| MatterSim | v1.0.0 1M/5M | approximately 1M–5M | [Model card](https://github.com/microsoft/mattersim/blob/main/MODEL_CARD.md) |
| MACE | MP-0, MP-0b/b2/b3, MPA-0, OMAT-0, MatPES PBE/r2SCAN, OFF23/OFF24, OMOL-0 (100M/4M training sets) | 0.69M–52.37M | [Foundation releases](https://github.com/ACEsuit/mace-foundations/releases), [OFF releases](https://github.com/ACEsuit/mace-off) |
| SevenNet | 0 (May/July 2024), l3i5, omat | 0.84M–approximately 15.8M | [Model descriptions](https://sevennet.readthedocs.io/en/latest/user_guide/pretrained.html) |
| NEP89 | `20250409` | approximately 0.976M | [Released model](https://github.com/brucefan1983/GPUMD/tree/master/potentials/nep/nep89_20250409) |

AISSquare publishes OMat weights for **both DPA4 and DPA4C**. The empty
DPA4C-OMat Hugging Face repository is not evidence that these weights are absent.
AISSquare also provides the matching training configurations and fine-tuning
commands. Use portable `.pt` checkpoints for training, not compiled `.pt2` exports.

### Fine-tuning prerequisites and exclusions

The September 2026 AIS Square releases are excluded: their model cards
require a development commit after 2026-10-01. The deployment recipe uses the
official DeepMD 3.2.0 GPU offline installer with matching August checkpoint/JSON
pairs. See [backend deployment](deployment/README.md) for the checkpoint catalog,
environment requirements, and validation steps. Validate each new installation;
catalog metadata does not establish GPU execution or support for every
checkpoint in TEMPER's training workflow.

MatPES release notes specify the main branch; those entries are
not certified for formal 3.2.0. Catalog availability alone does not establish
backend compatibility or install dependencies.
The catalog excludes entries with known unsupported TEMPER training workflows:
DeepMD OMol charge/spin conditioning, DPA4-Beta branch selection, SevenNet
multi-task modality assignment, MACE-MH head coordination, and PolarMACE's
specialized bundle. SevenNet-Nano is also excluded because its stated 0.13.1
requirement differs from TEMPER's 0.13.0 integration. These exclusions describe
TEMPER's present integration, not a claim that the upstream architectures cannot
be fine-tuned. MACE-MDP is excluded because it predicts dipoles/polarizabilities
rather than the benchmark's energies and forces.

Molecular checkpoints require appropriate charge/spin labels and native training
options. Consult the release notes and `usage_notes`; catalog selection cannot
supply dataset-dependent physical inputs. Source verification and specification
tests do not certify GPU training or evaluation with the installed backend.

MACE counts were obtained by summing parameter tensor shapes in the linked
release checkpoints, counting shared tensors once without executing checkpoint
code. SevenNet-0 and l3i5 use the model authors' counts recorded in
[Matbench Discovery](https://github.com/janosh/matbench-discovery/tree/main/models/sevennet);
SevenNet-omat is rounded from model-state tensor shapes. NEP89's estimate is
calculated from its published NEP4 architecture. MatterSim uses nominal sizes.

Custom checkpoint initialization remains available for integrations the caller
has validated. There is no moving `latest` alias: add a new variant key when a
checkpoint changes rather than silently replacing an existing selection.

Source filenames can change without changing the fixed names in submit folders.
Subclasses share `BaseSpecBuilder`; runtime adapters share `BaseMLFFAdapter`
under `temper.mlff.pipeline.training_adapters`. Both have `register(name, alias=None)`
decorators (alias accepts one string or a list of strings) and factories (`mlff_spec_builder_factory`, `mlff_adapter_factory`).
One generic `MLFFBundleWriter` packages all model families.
The public `temper.mlff.bundle_writers.write_submit_folder(bundle, target_dir)`
is also available through the bundle's method. Both return `(directory, packaged_bundle)`;
the returned record is independent of the source and matches `bundle.json`.

Builders read and hash every file immediately. The resolved absolute source
path is not part of MLFF identity, but each artifact key and SHA-256 digest is.
The writer checks the digest again immediately before copying, so changing a
file after `build()` is an error.

`training_parameters` and `testing_parameters` are ordinary mutable
dictionaries. Training parameters are overlaid on package defaults. Testing
parameters are passed directly to the selected ASE Calculator. Do not include
a `device` key in either dictionary: the submit host, not the machine creating
the bundle, determines hardware.

Builders own training policy. Runtime adapters add packaged input paths and
disable stress losses when labels are absent; they preserve bundle overrides.
Nested dictionaries are replaced as complete native sections.

| Family | Epoch key | Default epochs |
| --- | --- | --- |
| DPA-4 | `numb_epoch` | 60 |
| DPA-4C | `numb_epoch` | 100 |
| MatterSim | `epochs` | 200 |
| MACE | `max_num_epochs` | 100 |
| SevenNet / TorchNEP | `epoch` | 100 |

Epoch counts must be positive integers. Early stopping is disabled or patience
is placed beyond the requested run in the builder.

DeepMD uses the complete ordered type map from the matching explicit model
JSON, independently of the species present in a downstream dataset. The two
tested August OMat24 pairs contain all 118 elements, H through Og; TEMPER does
not impose that map on other checkpoints. Its descriptor and fitting network
come from the checkpoint via `--use-pretrain-script`. DeepMD 3.2.0 follows the
[fine-tuning rules](https://github.com/deepmodeling/deepmd-kit/blob/v3.2.0/deepmd/utils/finetune.py),
which require the target type map and remap pretrained types to it. The flag
does not replace every outer model setting, so preserve the matching model
section. `training_parameters` contains native training keys, plus `model`,
`optimizer`, `loss` and `learning_rate` dictionaries that the adapter places at the
top level of input.json. Defaults use HybridMuon (weight decay 0.001), gradient
norm 1, MAE normalized-force loss, energy/force weights 20, virial weight 5,
seed 42, save frequency 400 and display frequency 100. Both data batches use
`auto:128`. The learning-rate defaults are exponential, 1e-4 to 1e-6.
The [DPA-4 release guidance](https://huggingface.co/deepmodelingcommunity/DPA4-OMat24#fine-tune-on-a-downstream-dataset)
recommends a 1e-4 starting rate; the ending rate and DPA-4C schedule are TEMPER
defaults, not independently verified release recommendations. The
[DPA-4C model card](https://www.aissquare.com/models/detail?pageType=models&id=433&name=DPA4C-OMat24)
provides the matching JSON. Override the complete `learning_rate` section
when a chosen release specifies another schedule.

DeepMD writes its adapted configuration to `training_outputs/train_adapted.json`
and its portable trained model to `training_outputs/model.ckpt.pt`. The frozen
`training_outputs/evaluation.pt2` is temporary and recorded separately as
`files.evaluation_model`. `save_ckpt` is the prefix `model.ckpt`: DeepMD appends
`.pt`. The runner creates the export immediately before evaluation and removes
it afterward, including when freezing or evaluation fails. On a different host,
the portable checkpoint can be frozen again for that host.

MACE follows [naive fine-tuning](https://mace-docs.readthedocs.io/en/latest/guide/finetuning.html#naive-fine-tuning):
one head, estimated E0s, energy/force weights 10, learning rate 0.001, no weight
decay, EMA 0.999, AMSGrad, gradient clipping 1 and seed 42. TEMPER defaults to
batch size 4 and float32 (supported by the pinned 0.3.16 parser). Stress uses
the native stress loss and weight 1; without stress labels the runtime adapter selects
the energy/force loss.

SevenNet uses `train.continue.checkpoint` to load local SevenNet-0 weights,
with fresh optimizer, scheduler and epoch counters. On the runner,
`temper.mlff.pipeline.data_preparation.sevennet` uses SevenNet's
[checkpoint YAML export](https://github.com/MDIL-SNU/SevenNet/blob/v0.13.0/sevenn/checkpoint.py)
to obtain the model architecture, then applies the bundle's trainability
settings. Native training reads `training_outputs/sevennet_resolved.yaml`.
The checkpoint supplies the architecture and weights, as in the
[fine-tuning tutorial](https://github.com/MDIL-SNU/sevennet_tutorial/blob/main/notebooks/SevenNet_finetune_tutorial.ipynb).
The [0.13.0 continuation implementation](https://github.com/MDIL-SNU/SevenNet/blob/v0.13.0/sevenn/scripts/processing_continue.py)
also restores the checkpoint's species and normalization statistics.

NEP-89 fine-tuning uses TorchNEP rather than GPUMD's SNES optimizer. Its
architecture is derived from the pretrained `nep89.txt`; callers cannot
override architecture fields. The original labeled extxyz files go directly to TorchNEP. Its
[1.0.2 reader](https://github.com/mushroomfire/torchnep/blob/v1.0.2/torchnep/data.py)
accepts ASE stress (nine components, eV/Angstrom^3) and computes virial internally;
no conversion file or extra copy is needed. Full periodicity and a validation
dataset remain required.

## Persisted schemas and identity

`MLFFSpec` stores:

- a plain string MLFF family key;
- pinned implementation names, versions, and kinds;
- the local pretrained-model record;
- `training_parameters`, a dictionary or `None`; and
- `testing_parameters`, a flat calculator dictionary.

Its deterministic ID includes all scientific settings, artifact keys, and
artifact hashes. It excludes machine-local artifact paths. The field renaming
uses the identity schema `temper.mlff-spec.v1`. `LocalArtifactRef` lives in
`temper.schemas.artifact`.

`MLFFTrainBundle` retains the original `TrainingUnit`, `MLFFSpec`, and their
deterministic bundle ID. Packaging adds `schema_version`, `temper_version`, and
`files`: the copied pretrained model, training inputs and outputs, test inputs
and outputs, and any temporary evaluation model. Every recorded path starts at
the directory containing bundle.json. Packaging does not change scientific
identity or mutate the original local object. Original source paths describe
where inputs came from; the runtime uses the copied paths exclusively.
Normal TrainingUnit construction reads original datasets to calculate missing
hashes or verify supplied hashes before finalizing identity. Remote loading and
metadata inspection use `context={"verify_files": False}` to preserve recorded
hashes and identities without accessing original source files.

## Submit-directory contract

`write_submit_folder()` returns the new directory and a separate packaged
bundle matching `bundle.json`. If no destination is supplied, it allocates a
readable model/version/domain/bundle/attempt directory below
`DEFAULT_BUNDLE_WRITE_ROOT` (default `./bundles`). The `bundle_write_root` keyword
overrides that root. An explicit target directory is still supported and must
not already exist. See [Remote jobs](remote-jobs.md) for the naming and retention
rules; ordinary packaging alone does not upload or remove its copied inputs.

The default layout over the course of a fine-tuning run is:

~~~text
bundle/
    bundle.json
    run.sh
    results.json                # planned tests, then execution measurements
    pretrained_models/
        <pretrained model>
    training_inputs/
        train.extxyz
        validation.extxyz       # optional
        <native configuration>
        run.sh                  # CLI backends only
        data/                   # converted DeepMD datasets, when needed
    training_outputs/
        training.log
        <fine-tuned model>
        <adapted configuration, checkpoints, and other backend outputs>
        evaluation.pt2          # temporary DeepMD export
    testing_inputs/
        test_own.extxyz         # original test filenames and subdirectories
    testing_outputs/
        test_own.npz
~~~

Packaging writes bundle.json, the root launcher, copied weights and datasets.
Runtime preparation writes native training configs and the training script for
CLI backends using the copied inputs. Repeated preparation preserves
existing files, including manual edits, and restores missing configs. Execution
produces the remaining outputs. A populated
`files` field describes planned locations, not a claim that every file exists.
Zero-shot bundles have no training inputs or training outputs and evaluate the
copied pretrained model directly.

No Python modules or schema definitions are copied: the installed package owns
both validation and execution. The folder may be moved to another host without
the original datasets or pretrained-model directory.

On the execution host, run either:

~~~console
temper_bench run_pipeline /path/to/submit/bundle.json
bash /path/to/submit/run.sh
~~~

To generate inspectable configs and prepare output directories without requiring
a GPU, run the following locally or on the execution host:

~~~console
temper_bench run_pipeline /path/to/submit/bundle.json --prepare-only
~~~

Preparation writes native files and `training_inputs/run.sh` for CLI-based training.
These are derived from
`bundle.json`; preparation fills missing files without overwriting existing ones.
TorchNEP training is called directly through Python and writes `training_outputs/training.log`;
it does not generate a training script. Internal pipeline modules have no CLI
entry points. Use `temper_bench run_pipeline` for execution or its `--prepare-only` option
to inspect configs, and `temper.mlff.pipeline.runner.run_pipeline` from Python.
Execution checks CUDA, calls backend preparation functions (DeepMD conversion or
SevenNet checkpoint resolution), runs native training when requested, then evaluates
the trained model (or pretrained model for zero-shot). Predictions go under
`testing_outputs/`; training logs and fine-tuned models go under `training_outputs/`. The runner records the
actual TEMPER version alongside backend package versions in result metadata.
NEP architecture and data checks now run during runtime preparation.

Directory defaults are configurable through the environment variables in
[Default variables](default_variables.md), captured when the folder is written.
Remote environment defaults do not override the serialized layout.

Bundle schema version 1 defines this layout. Experiment IDs do not change merely because their
file layout changes. Format versions and TEMPER
release versions are separate. A TEMPER release mismatch emits a warning and
continues; an unsupported schema version still fails validation.

## Labels and automatic stress use

`TrainingUnit.dataset_stress` lazily inspects and caches each referenced file.
`training_has_stress` and `test_has_stress` expose that information to the
writer, which records it in the bundle. Reassigning dataset references or the root invalidates
the cache; exported files are treated as immutable after inspection.
The shared extxyz utility reads frame headers without constructing ASE Atoms.
InfoEntry additionally reads species columns for formulas; NEP uses those
columns to check model coverage and header periodicity. Numerical atom data is
not parsed during metadata inspection. Virial is detected separately and does
not imply an ASE stress label. The inspection checks:

- energy and forces must be present on every frame;
- a dataset in which every frame has stress enables stress prediction and, for
  train/validation data, the package's native stress or virial loss;
- a dataset with no stress omits that property; and
- mixed stress availability within one dataset is rejected.

For fine-tuning, train and validation datasets must agree about stress
availability. Test datasets are handled independently, so one can request
stress while another omits it without exposing a property-selection setting.

The evaluator preserves frame order and writes one compressed NumPy file per
test dataset:

| Array | Shape | ASE units |
| --- | --- | --- |
| `energies` | `(n_frames,)` | eV |
| `forces` | `(total_atoms, 3)` | eV/Angstrom |
| `atom_offsets` | `(n_frames + 1,)` | frame boundaries |
| `frame_indices` | `(n_frames,)` | original zero-based order |
| `stresses` | `(n_frames, 3, 3)`, when labels support stress | eV/Angstrom^3 |

The runner serializes a `MLFFExperimentManifest` into `results.json` at the calculation
root. It records training state and duration, references `bundle.json`, and
contains `MLFFTestManifest` entries with original names, states, paths, checksums,
requested properties, units and evaluation durations. Package versions are
recorded once in the experiment manifest. Each prediction NPZ contains only the
numerical arrays listed above; no embedded JSON, object arrays or pickle are used.
Frame and atom counts are obtained from the arrays when loading.
There are no new per-dataset metadata, summary or training-timing JSON files.

Test configurations and result manifests use schema
version 1. Evaluation checks the recorded bundle identity and dataset hashes.
Other schema versions are rejected; there are no format migration paths.

`files.result_manifest` is `"results.json"` in new fine-tuning and zero-shot
bundles. The runner creates it before preparation, saves
`training_wall_time_seconds` immediately after `adapter.train()` succeeds,
and updates it after every completed test. Training time includes loading,
validation and saving inside that call, but excludes separate preparation,
evaluation, scheduling and transfers. Failed training and zero-shot evaluation
have no completed training duration. A later evaluation failure preserves the
duration and all completed test entries.

Managed collection downloads this file and its referenced `bundle.json` unchanged.
It verifies settings and prediction checksums before cleanup. Local analysis reads
reference labels from the original split datasets; no reference archive is created.
Users must keep these datasets unchanged at their recorded original locations for
as long as results need to be analyzed. Each original test filename remains
separate, even when files have identical bytes. Callers retain their split-group
records separately.

```python
from temper.schemas import MLFFExperimentManifest

manifest = MLFFExperimentManifest.from_json_file("calculation/results.json")
experiment = manifest.load_results(base_dir="calculation")
for result in experiment.test_results:
    print(result.manifest.source_filename, result.predictions["energies"])
```

`MLFFExperimentManifest` and its `MLFFTestManifest` entries inherit from `MSONableModel`.
`from_json_file()` and `to_json_file()` use field dictionaries and the shared
atomic JSON writer. `load_results()` returns a
separate `MLFFExperimentResult` with the settings, measured training time and
loaded scientific data. It never adds arrays to the manifest. The runner and
collection code persist the manifest without assembling loaded experiment data.

See [Result loading](results-loading.md) for loading all completed test datasets,
moving result folders and retaining original reference datasets.

## Pinned integrations and remote device behavior

All training and evaluation requires a visible NVIDIA CUDA GPU. The shared
preflight runs before either stage and respects `CUDA_VISIBLE_DEVICES`.
There is no CPU or MPS fallback. Calculator construction checks CUDA as well;
DeepMD's native `DEVICE` environment setting is set to `cuda`, while
MACE, MatterSim and SevenNet receive a CUDA device explicitly.
NEP evaluation also requires the `gpumd` executable.
To select one GPU for the entire pipeline, run e.g.
`CUDA_VISIBLE_DEVICES=2 bash run.sh`; the selected physical GPU becomes logical
`cuda:0`. Preserve scheduler-provided visibility when running scheduled jobs.

| TEMPER key | Pinned implementation |
| --- | --- |
| `dpa4`, `dpa4c` | Official DeepMD-kit 3.2.0 offline installer for the recorded August AIS Square checkpoint/JSON pairs |
| `mattersim` | MatterSim 1.2.5 |
| `mace` | mace-torch 0.3.16 |
| `sevennet` | SevenNet 0.13.0 |
| `nep89` | TorchNEP 1.0.2, GPUMD 5.7, calorine 4.1 |

MatterSim defaults to 200 epochs, seed 42, learning rate 2e-4 and scheduler
step size 10. As checked on 2026-10-27, 1.2.5 remains the latest
[PyPI release](https://pypi.org/project/mattersim/), and its
[CUDA device and batched-index issue](https://github.com/microsoft/mattersim/issues/163)
requires the audited dependency fixes in
[backend deployment](deployment/README.md). Batch size still
defaults to 1, but the runtime adapter does not force this value: patched
installations can set `training_parameters={"batch_size": 4}`. Validation must
use at least two frames per batch. Version 1.2.5 alone does not establish
whether the fixes are installed; TEMPER no longer declares every such
installation defective. Stock 1.2.5 is not certified for GPU fine-tuning.

The submit environment for NEP fine-tuning must provide exactly TorchNEP
1.0.2 together with a compatible PyTorch and NumPy installation. TorchNEP is
not a core TEMPER dependency, and zero-shot NEP bundles do not require it.

## Deliberate boundary

Local regression tests can be run with `pip install -e ".[preprocess,test]"` followed
by `python -m pytest -q`. PyYAML tests the SevenNet preparation helper; the
remote SevenNet installation already provides it. GPU training remains an
integration check in the pinned runner environment.

The generated folder is local output. Remote upload, hosts, scheduler settings,
submission, polling, retries, result download, and benchmark metric aggregation
remain separate execution-layer concerns.

## File records and execution responsibilities

The populated `files` field describes these paths, all relative to the bundle root:

| Field | Contents |
| --- | --- |
| `pretrained_model` | Copied model used to start fine-tuning or to evaluate zero-shot |
| `training_input_files` | `train`, optional `validation`, native `config`, CLI `script`, and any converted datasets |
| `training_output_files` | `trained_model`, `training_log`, and backend-specific output files or directories |
| `test_input_files` | Original test filenames mapped to their copied paths under `testing_inputs/` |
| `test_output_files` | The same original filenames mapped to corresponding NPZ paths under `testing_outputs/` |
| `result_manifest` | `results.json` at the calculation root |
| `evaluation_model` | Temporary model export, or None when no export is needed |

Backend output entries include DeepMD's adapted configuration and learning curve;
MACE's checkpoints, logs, and results directories; SevenNet's resolved configuration
and native work directory; and MatterSim/TorchNEP native output directories.
Directory entries identify where backend-generated files appear. These fields do
not define download selection or implement a download manager.

Spec builders copy their backend-specific filename defaults into the required
`MLFFSpec.file_layout` field, an `MLFFFileLayout` record. It stores
`pretrained_model_filename`, `training_input_filenames`,
`training_output_filenames`, and optional `evaluation_model_filename`. Names may
contain subdirectories and are relative to the writer's corresponding model,
training-input, or training-output directory. For example, `mace.yaml` becomes
`training_inputs/mace.yaml` in `files.training_input_files["config"]`.

`create_bundles` saves these settings even though `bundle.files` is still None.
`MLFFBundleWriter(bundle)` later reads the saved layout, copies source inputs, and
records the resolved paths. It never consults a builder or adapter registry.
Changing installed defaults therefore cannot change a previously saved
specification's filename choices. Source checkpoints remain identified separately
by `MLFFSpec.pretrained_model.artifacts`.

Python callers can supply a complete `MLFFFileLayout` through a spec builder's
`file_layout` argument, or edit `spec.file_layout` before packaging. Builders copy
the layout so specifications do not share mutable filename dictionaries. Packaging
names do not affect scientific specification or bundle IDs. Once a bundle has
been packaged, further attempts must use the same file locations; changing them
is rejected. A filename change alone also does not create a new experiment in
`create_bundles`, which skips existing bundle IDs.

The runner selects an adapter from its installed registry and instantiates it
with the packaged bundle and root. Adapters return configs and
scripts through `training_file_contents(training_stress)` without writing them.
The standalone `pipeline.runner.write_configuration_files()` function writes these
files during preparation. NEP rendering reads the copied model header
and validates dataset species and periodicity; no training backend is imported.
During execution adapters use recorded paths for dataset conversion, checkpoint
resolution, training, and temporary model export and cleanup.

`prepare_bundle()` preserves existing configs. After deliberately changing a
packaged bundle or its recorded paths, regenerate derived configs explicitly with
`temper.mlff.pipeline.runner.write_configuration_files(adapter)` using an adapter bound to
that updated bundle and its root. Removing selected configs also lets preparation
restore just those files. Rewriting configs replaces manual edits; moving the
whole folder without changing its relative layout needs no regeneration.
TorchNEP trains directly through its Python API. Fine-tuned models are found at
`files.training_output_files["trained_model"]`, not by searching a directory.

Evaluation uses one `build_calculator()` function in
`temper.mlff.pipeline.calculators`. It imports only the selected backend.
Adding a new calculator requires adding a branch to that function. The evaluator
receives the bundle root explicitly, so changing the working directory does not
change how dataset and model paths are interpreted. `temper.mlff.pipeline.runner.make_test_manifests(bundle, ...)`
creates pending `MLFFTestManifest` entries containing execution input paths and reference checksums,
requested properties, units, and planned prediction paths. For older local bundles
without recorded dataset hashes, pass `bundle_root=bundle_directory` to calculate
them from the copied inputs.

The runner embeds these entries in an `MLFFExperimentManifest` and writes
`results.json` before execution. It calls
`run_test(manifest, bundle_root=bundle_directory)` after training. Evaluation reads
shared calculator settings and model locations from the manifest's referenced
bundle, uses each test manifest directly, and saves checksums, durations and status
after each test. Completed tests and training measurements survive a later failure.
Execution input and prediction paths are relative to `bundle_directory`. Local
analysis instead reads original labels from
`bundle.training_unit.root_path / test.source_domain / test.source_filename`.
Packaging records the original split-data root as an absolute path, so result
folders can move without changing where labels are read.

There is no generated `test_config.json`. Change shared evaluation settings in
the bundle's `mlff_spec.testing_parameters` before starting a run. The previous
`make_test_config()` and `run_test(config_path, ...)` APIs have been replaced by
the manifest APIs above. Manifests using removed reference-archive fields must be
regenerated; there is no automatic migration.
Old test-config files, if present in a previously prepared folder, are ignored.

All applicable default random seeds are 42, including SevenNet and TorchNEP's
Python API `run_seed`. MACE uses the supplied validation file when present;
TEMPER does not set `valid_fraction`. Without a validation file, MACE's native
validation-split default applies.

INFO messages identify the bundle being packaged, prepared, trained, or evaluated.
Backend training output is written to the recorded training.log rather than echoed
to the terminal. Failed training reports the log's location and stops before
evaluation. `temper_bench --verbose run_pipeline bundle.json` adds DEBUG file and
adapter details; `--quiet` suppresses routine progress. Python callers can
configure the `temper` logger with the standard logging module.


### Explicit DeepMD model configurations

For DPA4 and DPA4C, the builder copies `model` from the matching upstream JSON
into the specification. This includes the full `type_map` in checkpoint order
and outer model options that `--use-pretrain-script` does not copy. The source
JSON is not needed on the execution machine because its model settings are
embedded in the bundle and participate in its scientific identity. Presets
are not supported. The JSON's training data paths and original training budget
are not reused; TEMPER supplies bundle paths and the requested fine-tuning
controls. Explicit model settings can also be supplied as
`training_parameters={"model": model_section, "numb_epoch": 10}`.

MACE label defaults are defined once in the spec builder: `energy_key=energy`,
`forces_key=forces`, and `stress_key=stress`. User overrides take precedence.
Regenerate older MACE bundles instead of relying on runtime YAML migration.
