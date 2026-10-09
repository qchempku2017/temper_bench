# MLFF specifications and submit bundles

[Back to the project overview](../README.md) · [Default variables](default_variables.md) · [Important concepts and schemas](important-concepts-and-schemas.md)

TEMPER supports DPA-4, DPA-4C, MatterSim, MACE, SevenNet, and NEP-89.
Local builders create model recipes; `MLFFBundleWriter` copies their inputs and
serializes `MLFFTrainBundle` as `bundle.json`. An installed TEMPER runtime reads
that record remotely, prepares backend configs, fine-tunes, and evaluates.
Upload, scheduling, and result download remain separate executor concerns.

Install `temper-bench[preprocess]` for local data splitting. Execution hosts
need the base `temper-bench` package plus their selected MLFF backend, using
preferably the same TEMPER release as the writer. Other releases warn on load.
Bash is required for native training.

## Building a specification

Every builder accepts a source directory and optional release filenames.
Omitting the directory uses `DEFAULT_MLFF_PRETRAINED_MODELS_DIR`, whose default
is `./pretrained_models`.

~~~python
from temper.mlff import MACESpecBuilder, MLFFTrainBundle

spec = MACESpecBuilder(
    # Optional; defaults to ./pretrained_models/mace.model.
    pretrained_model_dir="/shared/models", model_filename="mace.model",
    # None means test the pretrained model without training.
    # An empty dict enables the pinned fine-tuning defaults.
    training_parameters={},
    # Flat native MACECalculator keyword arguments.
    testing_parameters={"default_dtype": "float32"},
).build()

bundle = MLFFTrainBundle(training_unit=training_unit, mlff_spec=spec)
submit_directory = bundle.write_submit_folder("submit/mace-run")
~~~

All builders accept `pretrained_model_dir` and `model_filename`.
No JSON sidecar is required. Default filenames are:

| Builder | Model filename |
| --- | --- |
| `DPA4SpecBuilder` | `dpa4.pt` |
| `DPA4CSpecBuilder` | `dpa4c.pt` |
| `MatterSimSpecBuilder` | `mattersim.pth` |
| `MACESpecBuilder` | `mace.model` |
| `SevenNetSpecBuilder` | `sevennet.pth` |
| `NEP89SpecBuilder` | `nep89.txt` |

Source filenames can change without changing the fixed names in submit folders.
Subclasses share `BaseSpecBuilder`; runtime adapters share `BaseMLFFAdapter`
under `temper.mlff.pipeline.training_adapters`. Both have `register(name, alias=None)`
decorators and factories (`mlff_spec_builder_factory`, `mlff_adapter_factory`).
One generic `MLFFBundleWriter` packages all model families.
The public `temper.mlff.bundle_writers.write_submit_folder(bundle, target_dir)`
is also available through the bundle's method.

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
disable stress losses when labels are absent; they preserve recipe overrides.
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

DeepMD always uses all 118 elements, H through Og in atomic-number order, as
its type map, independently of the species present in a dataset;
its descriptor and fitting network come from the checkpoint via
`--use-pretrain-script`. Both pinned backends use the same
[fine-tuning rules](https://github.com/deepmodeling/deepmd-kit/blob/v3.2.0/deepmd/utils/finetune.py),
which require the target type map and remap pretrained types to it. `training_parameters` contains native training keys,
plus `loss` and `learning_rate` dictionaries that the runtime adapter places at the
top level of input.json. Defaults use HybridMuon (weight decay 0.001), gradient
norm 1, MAE normalized-force loss, energy/force weights 20, virial weight 5,
seed 42, save frequency 400 and display frequency 100. Both data batches use
`auto:128`. The learning-rate defaults are exponential, 1e-4 to 1e-6.
The [DPA-4 release guidance](https://huggingface.co/deepmodelingcommunity/DPA4-OMat24#fine-tune-on-a-downstream-dataset)
recommends a 1e-4 starting rate; the ending rate and DPA-4C schedule are TEMPER
defaults, not independently verified release recommendations. The
[DPA-4C model card](https://huggingface.co/deepmodelingcommunity/DPA4C-OMat)
currently provides no recipe. Override the complete `learning_rate` section
when a chosen release specifies another schedule.

DeepMD writes its adapted input to `outputs/train_adapted.json` and retains
both `artifacts/model.ckpt.pt` and the frozen `artifacts/dpa4.pt2` or
`dpa4c.pt2`. `save_ckpt` is the prefix `model.ckpt`: DeepMD appends `.pt`.
Refreeze the portable checkpoint on a different target machine.

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
to obtain the model architecture, then applies the recipe's trainability
settings. Native training reads `training/sevennet_resolved.yaml`.
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
changes the identity schema to `temper.mlff-spec.v3`; rebuild old specifications
and their bundles. There are no legacy aliases. `LocalArtifactRef` now lives in
`temper.schemas.artifact`.

`MLFFTrainBundle` retains the original `TrainingUnit`, `MLFFSpec`, and their
deterministic bundle ID. Packaging adds `schema_version`, `temper_version`, and
`files`: a bundle-relative dataset/model mapping, resolved stress availability,
the trained-model filename, and training/artifact/output directories. Packaging does not change scientific
identity or mutate the original local object. Original source paths are
provenance; the runtime uses the packaged mapping exclusively. TrainingUnit
loading validates metadata without accessing source files; local file access
checks existence when needed.

## Submit-directory contract

`write_submit_folder()` copies ordinary files into a new directory. If omitted,
the destination is a caller-owned temporary directory. The initial folder is:

~~~text
submit/
├── bundle.json
├── run.sh
├── datasets/
│   ├── train.extxyz          # fine-tuning only
│   ├── validation.extxyz     # when provided
│   └── test_000.extxyz       # one per test dataset
└── models/
    └── <pretrained files>
~~~

No Python modules or schema definitions are copied: the installed package owns
both validation and execution. The folder may be moved to another host without
the original datasets or pretrained-model directory.

On the execution host, run either:

~~~console
temper_bench run_pipeline /path/to/submit/bundle.json
bash /path/to/submit/run.sh
~~~

To inspect native files without starting training or requiring a GPU:

~~~console
temper_bench run_pipeline /path/to/submit/bundle.json --prepare-only
~~~

Preparation writes native files and `training/run.sh` for CLI-based training, plus
`test_config.json` for evaluation. These are derived outputs of `bundle.json`.
TorchNEP training is called directly through Python and writes `outputs/training.log`;
it does not generate a training script. Internal pipeline modules have no CLI
entry points. Use `temper_bench run_pipeline` for execution or its `--prepare-only` option
to inspect configs, and `temper.mlff.pipeline.runner.run_pipeline` from Python.
Execution checks CUDA, calls backend preparation functions (DeepMD conversion or
SevenNet checkpoint resolution), runs native training when requested, then evaluates
the trained model (or pretrained model for zero-shot). Predictions and logs go
under `outputs/`; trained models go under `artifacts/`. The runner records the
actual TEMPER version alongside backend package versions in result metadata.
NEP architecture and data checks now run during runtime preparation.

Directory defaults are configurable through the environment variables in
[Default variables](default_variables.md), captured when the folder is written.
Remote environment defaults do not override the serialized layout.

This format replaces the old copied-runtime folder contract. Regenerate old
submit folders for the installed-runtime workflow. Format versions and TEMPER
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

Adjacent JSON files serialize `temper.schemas.MLFFTestResult`, recording source
metadata, package versions, units, array counts and wall time. Use
`MLFFTestResult.from_dict(json.loads(path.read_text()))` to load a result.
The installed package supplies the result schema. `outputs/test_summary.json` lists all evaluated datasets.

## Pinned integrations and remote device behavior

All training and evaluation requires a visible NVIDIA CUDA GPU. The shared
preflight runs before either stage and respects `CUDA_VISIBLE_DEVICES`.
There is no CPU or MPS fallback. Calculator adapters check CUDA as well;
DeepMD's native `DEVICE` environment setting is set to `cuda`, while
MACE, MatterSim and SevenNet receive a CUDA device explicitly.
NEP evaluation also requires the `gpumd` executable.
To select one GPU for the entire pipeline, run e.g.
`CUDA_VISIBLE_DEVICES=2 bash run.sh`; the selected physical GPU becomes logical
`cuda:0`. Preserve scheduler-provided visibility when running scheduled jobs.

| TEMPER key | Pinned implementation |
| --- | --- |
| `dpa4`, `dpa4c` | DeepMD-kit 3.2.0 |
| `mattersim` | MatterSim 1.2.5 |
| `mace` | mace-torch 0.3.16 |
| `sevennet` | SevenNet 0.13.0 |
| `nep89` | TorchNEP 1.0.2, GPUMD 5.7, calorine 3.5 |

MatterSim defaults to 200 epochs, seed 42, learning rate 2e-4 and scheduler
step size 10. As checked on 2026-10-08, 1.2.5 remains the latest
[PyPI release](https://pypi.org/project/mattersim/), and its
[CUDA device and batched-index issue](https://github.com/microsoft/mattersim/issues/163)
remains open. Batch size therefore still defaults to 1 and training emits a
warning. The runtime adapter does not force this value: patched installations can
set `training_parameters={"batch_size": 4}`. This does not claim that stock
1.2.5 GPU fine-tuning is fixed.

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
