# MLFF specifications and submit bundles

[Back to the project overview](../README.md) · [Default variables](default_variables.md) · [Important concepts and schemas](important-concepts-and-schemas.md)

TEMPER supports six MLFF families: DPA-4, DPA-4C, MatterSim, MACE,
SevenNet, and NEP-89. The local layer does only two things:

1. A family builder hashes local pretrained files and creates an `MLFFSpec`.
2. `MLFFTrainBundle` pairs that recipe with one `TrainingUnit` and
   `write_submit_folder()` copies a self-contained directory.

It does not download models, choose local hardware, submit a scheduler job, or
run third-party training.

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
Subclasses share `BaseSpecBuilder`; adding a release normally means overriding
class metadata, filenames and defaults. Builders and writers register using
`@BaseSpecBuilder.register(name="family", alias="short_name")` and
`@BaseMLFFBundleWriter.register(name="family", alias="short_name")`.
The alias is optional; registry keys are independent of class attributes.
`mlff_spec_builder_factory(key)` and `mlff_bundle_writer_factory(key)` return classes.
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

Builders own all training policy. Writers add dataset and artifact paths and
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
plus `loss` and `learning_rate` dictionaries that the writer places at the
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
the native stress loss and weight 1; without stress labels the writer selects
the energy/force loss.

SevenNet uses `train.continue.checkpoint` to load local SevenNet-0 weights,
with fresh optimizer, scheduler and epoch counters. On the runner,
`prepare_sevennet.py` (copied from
`runtime/data_preparation/sevennet.py`) uses SevenNet's
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

`MLFFTrainBundle` stores only one `TrainingUnit`, one `MLFFSpec`, and its
deterministic ID. Its `unit_type` property comes from
`TrainingUnit`. There are no capability records, input/output manifests, layout
objects, named-model resolvers, or copy-mode settings.

## Submit-directory contract

`write_submit_folder()` always makes ordinary file copies. The destination must
not exist; when omitted, TEMPER creates and returns a caller-owned temporary
directory. A zero-shot directory contains:

~~~text
submit/
├── run.sh
├── test_config.json
├── datasets/
│   ├── test_000.extxyz
│   └── test_001.extxyz
├── models/
│   └── <fixed pretrained files>
└── runtime/
    ├── run_test.py
    ├── calculator.py
    ├── check_cuda.py
    └── result_schema.py
~~~

A fine-tuning directory additionally contains `train.extxyz`, an optional
`validation.extxyz`, package-native files below the training directory, and
later writes its trained model below the artifacts directory. MACE, MatterSim,
SevenNet and NEP use `finetuned_` filenames to distinguish trained outputs from
pretrained inputs. `run.sh` contains
no scheduler directives. It runs native training when required and then invokes
the common ASE evaluator.

The six submit subdirectory names are configurable only through the environment
variables documented in [Default variables](default_variables.md). Individual
filenames and command templates are fixed.

## Labels and automatic stress use

`TrainingUnit.dataset_stress` lazily inspects and caches each referenced file.
`training_has_stress` and `test_has_stress` expose that information to all
writers sharing the unit. Reassigning dataset references or the root invalidates
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
The same standard-library schema is copied as `runtime/result_schema.py`. `outputs/test_summary.json` lists all evaluated datasets.

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
warning. The writer no longer forces this value: patched installations can
set `training_parameters={"batch_size": 4}`. This does not claim that stock
1.2.5 GPU fine-tuning is fixed.

The submit environment for NEP fine-tuning must provide exactly TorchNEP
1.0.2 together with a compatible PyTorch and NumPy installation. TorchNEP is
not a core TEMPER dependency, and zero-shot NEP bundles do not require it.

## Deliberate boundary

Local regression tests can be run with `pip install -e ".[test]"` followed
by `python -m pytest -q`. PyYAML tests the SevenNet preparation helper; the
remote SevenNet installation already provides it. GPU training remains an
integration check in the pinned runner environment.

The generated folder is local output. Remote upload, hosts, scheduler settings,
submission, polling, retries, result download, and benchmark metric aggregation
remain separate execution-layer concerns.
