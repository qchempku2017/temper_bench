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

Every builder accepts optional paths to local pretrained files. Omitting a path
uses a fixed filename below `DEFAULT_MLFF_PRETRAINED_MODELS_DIR`, whose default
is `./pretrained_models`.

~~~python
from temper.mlff import MACESpecBuilder, MLFFTrainBundle

spec = MACESpecBuilder(
    # Optional; defaults to ./pretrained_models/mace.model.
    pretrained_model_path="/shared/models/mace.model",
    # None means test the pretrained model without training.
    # An empty dict enables the pinned fine-tuning defaults.
    training_parameters={},
    # Flat native MACECalculator keyword arguments.
    testing_parameters={"default_dtype": "float64"},
).build()

bundle = MLFFTrainBundle(training_unit=training_unit, mlff_spec=spec)
submit_directory = bundle.write_submit_folder("submit/mace-run")
~~~

The accepted model-path arguments and fixed submit filenames are:

| Builder | Optional source-path arguments | Files copied below the models directory |
| --- | --- | --- |
| `DPA4SpecBuilder` | `pretrained_model_path`, `pretrained_config_path` | `dpa4.pt`, `dpa4.json` |
| `DPA4CSpecBuilder` | `pretrained_model_path`, `pretrained_config_path` | `dpa4c.pt`, `dpa4c.json` |
| `MatterSimSpecBuilder` | `pretrained_model_path` | `mattersim.pth` |
| `MACESpecBuilder` | `pretrained_model_path` | `mace.model` |
| `SevenNetSpecBuilder` | `pretrained_model_path` | `sevennet.pth` |
| `NEP89SpecBuilder` | `pretrained_model_path` | `nep89.txt` |

Builders read and hash every file immediately. The resolved absolute source
path is not part of MLFF identity, but each artifact key and SHA-256 digest is.
The writer checks the digest again immediately before copying, so changing a
file after `build()` is an error.

`training_parameters` and `testing_parameters` are ordinary mutable
dictionaries. Training parameters are overlaid on package defaults. Testing
parameters are passed directly to the selected ASE Calculator. Do not include
a `device` key in either dictionary: the submit host, not the machine creating
the bundle, determines hardware.

Every gradient-based integration uses a native epoch count and defaults to 100
epochs. DeepMD uses `numb_epoch`, MACE uses `max_num_epochs`, MatterSim uses
`epochs`, and SevenNet and TorchNEP use `epoch`. Counts must be positive
integers. TEMPER disables or places early stopping beyond the configured run so
all requested epochs complete; training failures and external interruption are
the only exceptions.

DeepMD's JSON sidecar remains required for its type map, loss, learning-rate,
and other non-architecture policy. The generated input deliberately leaves
`descriptor` and `fitting_net` empty, invokes `dp train` with
`--use-pretrain-script`, and writes DeepMD's fully adapted input to
`outputs/train_adapted.json`.

NEP-89 fine-tuning uses TorchNEP rather than GPUMD's SNES optimizer. Its
architecture is derived from the pretrained `nep89.txt`; callers cannot
override architecture fields. Persisted NEP specifications that contain a
`restart` artifact or the old `generation`, `population`, `save_potential`, or
`lambda_1` controls must be rebuilt.

## Persisted schemas and identity

`MLFFSpec` stores:

- a plain string MLFF family key;
- pinned implementation names, versions, and kinds;
- the local pretrained-model record;
- a training dictionary or `None`; and
- a flat testing dictionary.

Its deterministic ID includes all scientific settings, artifact keys, and
artifact hashes. It excludes machine-local artifact paths.

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
    └── device.py        # only when the integration needs it
~~~

A fine-tuning directory additionally contains `train.extxyz`, an optional
`validation.extxyz`, package-native files below the training directory, and
later writes its trained model below the artifacts directory. `run.sh` contains
no scheduler directives. It runs native training when required and then invokes
the common ASE evaluator.

The six submit subdirectory names are configurable only through the environment
variables documented in [Default variables](default_variables.md). Individual
filenames and command templates are fixed.

## Labels and automatic stress use

Before copying, TEMPER reads every referenced frame:

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

Adjacent JSON files record source metadata, package versions, units, and wall
time. `outputs/test_summary.json` lists all evaluated datasets.

## Pinned integrations and remote device behavior

| TEMPER key | Pinned implementation | Training and testing device behavior |
| --- | --- | --- |
| `dpa4` | DeepMD-kit 3.2.0 | DeepMD chooses CPU or CUDA natively for training and Calculator evaluation. |
| `dpa4c` | DeepMD-kit 3.2.0 | Same native DeepMD selection, using the PyTorch-exportable CLI profile. |
| `mattersim` | MatterSim 1.2.5 | Training resolves CUDA or CPU on the submit host; Calculator evaluation omits device and uses MatterSim's native choice. Batch size is fixed to one, and CUDA training prints a warning for the known 1.2.5 issue. |
| `mace` | mace-torch 0.3.16 | TEMPER resolves CUDA, then MPS, then CPU on the submit host for training and Calculator evaluation. |
| `sevennet` | SevenNet 0.13.0 | Device is omitted; SevenNet's native auto mode selects CUDA or CPU. |
| `nep89` | TorchNEP 1.0.2, GPUMD 5.7, and calorine 3.5 | Fine-tuning performs a CUDA preflight, starts a fresh gradient-descent run from `nep89.txt`, and publishes the lowest-validation-loss `nep_best.txt`. Evaluation selects `GPUNEP` only when CUDA and the `gpumd` executable are visible; otherwise it uses `CPUNEP`. |

Resolution happens only inside the written runtime. Scheduler-provided
`CUDA_VISIBLE_DEVICES` restrictions are honored. TEMPER does not retry on a
different device after a third-party package has started.

The submit environment for NEP fine-tuning must provide exactly TorchNEP
1.0.2 together with a compatible PyTorch and NumPy installation. TorchNEP is
not a core TEMPER dependency, and zero-shot NEP bundles do not require it.

## Deliberate boundary

The generated folder is local output. Remote upload, hosts, scheduler settings,
submission, polling, retries, result download, and benchmark metric aggregation
remain separate execution-layer concerns.
