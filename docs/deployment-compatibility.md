# Submission and backend compatibility

TEMPER's submission host and GPU execution environments have different dependencies.
Install `temper-bench[remote,preprocess]` on native Windows, Linux or WSL. Install TEMPER and
one model backend per Linux GPU environment. A working import or `pip check`
does not establish that fine-tuning, export or inference works.

## Submission host

DPDispatcher remains pinned to 1.0.3. The 1.1.0 release was inspected on
2026-10-27: `SSHContext._put_files` still converts the local archive path with
`as_posix()`. Upgrading alone therefore does not repair extended Windows paths.
TEMPER's verified SSH context now archives its explicit forward files using
native normalized paths; only archive member names and remote paths use POSIX
notation. Downloads use native normalized destinations and are checked against
remote SHA256 values before atomic installation. No installed package is edited.
WSL uses Linux paths, including its own key/known_hosts paths.

## Supported execution recipes

| Family | Execution dependencies | Compatibility treatment |
| --- | --- | --- |
| MACE | mace-torch 0.3.16, e3nn 0.4.4 | Build the local-checkpoint dtype patch below |
| MatterSim | MatterSim 1.2.5 | Build both device-transfer and three-body-index fixes below |
| NEP89 | TorchNEP 1.0.2, GPUMD 5.7, calorine 4.1 | Use the upstream calorine release; no calorine source patch |
| SevenNet | sevenn 0.13.0 | No additional compatibility patch |
| DPA4 / DPA4C | Official DeepMD 3.2.0 GPU offline installer | Explicit checkpoint/JSON pairs; validation status below |

The MACE/MatterSim latest PyPI releases still contained the affected code when
checked. Build patches into wheels instead of editing site-packages:

```bash
# In an isolated build environment, with Git available:
python -m pip install build
python compatibility/build_backend.py mace-torch wheelhouse
python compatibility/build_backend.py mattersim wheelhouse
# In each matching execution environment, use normal dependency resolution:
python -m pip install wheelhouse/mace_torch-0.3.16+temper.1-*.whl
# Or the MatterSim wheel in its separate environment:
python -m pip install wheelhouse/mattersim-1.2.5+temper.1-*.whl
python -m pip check
```

The Linux Python 3.12 recipes in `compatibility/constraints/` pin the core
versions from the audited environments. Install CUDA PyTorch from the official
`https://download.pytorch.org/whl/cu128` index, then pass the matching constraints
file when installing TEMPER and the backend wheel, for example
`python -m pip install -c compatibility/constraints/mace.txt TEMPER_WHEEL MACE_WHEEL`.
For NEP use `calorine==4.1` and `torchnep==1.0.2` with the nep89 constraints;
build/install GPUMD 5.7 for the actual GPU separately. DeepMD retains its dedicated
official offline-installer workflow. These are execution-host recipes, not
dependencies to install on a native Windows submission host. Complete historical
inventories remain in `docs/deployment/audit-*.json`.

The builder verifies the upstream source hash, checks patch applicability, builds
a local-version wheel, and writes source/patch/wheel hashes beside it. Use
`--archive /path/to/verified-source.tar.gz` for a workstation-relayed download.
Build MatterSim on the target Linux/Python architecture. Installation still
requires the matching CUDA PyTorch stack and backend dependencies. Keep the
build JSON with the environment inventory. Never use `pip --no-deps` or ignore
resolver errors as an installation strategy.

For the recorded checkpoint/configuration pairs, run
`python compatibility/check_installation.py FAMILY CHECKPOINT result.json`
after activating the execution environment. This checks checkpoint SHA256,
exact recipe versions, required patch source hashes, executable availability
and `pip check`, without importing models. An unknown checkpoint is rejected
because its backend requirements have not been established. This check does
not execute GPUMD to discover its version; retain its build provenance and
binary hash separately, then verify it in a scheduled GPU test.

The [29 October August-checkpoint matrix](deployment/august-validation-20261029.md)
adds native single-GPU fine-tuning, freeze, ASE inference and export-cleanup
results for all 16 AIS Square DPA4/DPA4C Nano-through-Air OMat24 and
OMol25-100M August pairs. All passed with the formal 3.2.0 offline
installation. Exact hashes, package versions and Slurm job IDs accompany
the report. These tiny tests establish execution compatibility; they do
not establish scientific accuracy or 16 TEMPER JobManager round trips.

MACE configuration now explicitly uses ASE's `energy`, `forces`, and `stress`
keys in new specifications. Explicit overrides remain intact. Existing MACE
bundles must be regenerated; the runtime does not rewrite old YAML files. The dependency patch casts the local checkpoint to
the requested dtype before energy-offset estimation. It does not change loss,
precision selection, epochs or other scientific controls.

MatterSim's graph converter must instantiate `M3GNetData` so PyG offsets
three-body *edge* indices by each frame's bond count. Its training loop must
move the entire batch, including labels, to the model device. A two-frame
training batch is required for validation. TEMPER does not infer installed fix
status from version 1.2.5 or emit a claim that every installation is defective.
Compare patched source hashes and run the scheduled test.

Calorine 4.1 uses `dump_xyz` for GPUMD 5.7 and no longer imposes the historical
NumPy ceiling. TEMPER explicitly supplies `gpu_identifier_index=None`, preserving
the scheduler's `CUDA_VISIBLE_DEVICES`; a caller's explicit override is retained.
Energy is in eV, forces in eV/angstrom, and ASE stress in eV/angstrom cubed.
TEMPER prediction arrays have shapes `(frames,)`, `(total_atoms, 3)` and
`(frames, 3, 3)` respectively. No unit conversion was changed by these fixes.

TEMPER has no need for an exact NumPy 2.5.2 pin. Its core requirement is now
`numpy>=2.0,<3`; the resolver can choose a version compatible with optional
preprocessing packages. The validation record identifies tested versions; this
range is not a claim that every backend/optional dependency supports every
NumPy release. Run `pip check` in each resolved environment.

## DeepMD checkpoint and schema requirements

Use the latest tested formal DeepMD release from its official GPU offline
installer, in a dedicated environment. The tested CUDA 12.9 installer
contains PyTorch 2.12.0. Add `vesin-torch==0.6.1`, ASE and dpdata using
normal dependency resolution and run `pip check`. See the
[installation and validation record](deployment/review-validation-20261027.md). Do not substitute a Conda-channel build
or upgrade it to a development revision to accommodate a newer checkpoint.
Choose compatible checkpoint and explicit JSON pairs from AIS Square instead.
The installer may use Conda activation internally; that does not require
resolving DeepMD packages from a Conda channel.

Freeze compiles a GPU inference library. Its linker needs an unversioned
`libcuda.so`, which this offline installer does not supply. An installed CUDA
toolkit normally provides a link-time stub in `lib64/stubs`. If freeze reports
`cannot find -lcuda`, verify that file and append its directory to
`LIBRARY_PATH` in the execution environment. Preserve any existing entries.
The directory is site-specific; it belongs in the resource configuration.
Do not add the stub directory to `LD_LIBRARY_PATH`: execution must load the
real NVIDIA driver, not the link-time placeholder. On WSL, the driver interface
is instead provided under `/usr/lib/wsl/lib`. GPU validation must still run
inside the allocation after correcting the link path.

For fine-tuning, place the matching JSON next to the checkpoint with the same
stem, for example `DPA4C-Air-OMat24-v20260819.pt` and
`DPA4C-Air-OMat24-v20260819.json`. The builder embeds its model section in the
specification. Alternatively supply `training_parameters.model` explicitly.
The complete checkpoint element list and its order must be retained. TEMPER
does not substitute all 118 elements for a checkpoint with a different map.
Preset-based JSON is rejected; use an explicit configuration file.

DeepMD 3.2.0's fine-tuning helper copies descriptor and fitting-network settings
when `--use-pretrain-script` is supplied, not every outer model setting. The
matching explicit model section therefore remains important. DPA4 accepts the
upstream `dpa4`, `DPA4`, `SeZM`, and `sezm` aliases. DPA4C uses the standard model
and a `dpa4c` descriptor. Optimizer settings belong at the top level. Legacy
bundles with `training.optimizer` are normalized without losing parameters;
equal duplicates collapse, conflicting values raise an error before writing.

The earlier [deployment evidence](deployment/validation-20261027.md) describes
September checkpoints tested with a development backend. It is historical
evidence, not formal-release acceptance. Record installer, checkpoint and JSON
hashes with each new test. Validate actual fine-tuning, freeze, ASE energy,
forces and stress, and export cleanup on WSL2 and through the remote scheduler.
Installation checks alone cannot certify a checkpoint/backend combination.

## Resources and recovery

New Slurm profiles can specify:

```yaml
resources:
  number_node: 1
  tasks_per_node: 1
  cpus_per_task: 4
  gpu_per_node: 1
  queue_name: your_gpu_partition
  custom_flags:
    - '#SBATCH --time=01:00:00'
slurm_preflight: true
```

This requests one node, one task, four CPUs per task and one GPU per node.
Do not also specify `cpu_per_node` or CPU/task directives in `custom_flags`.
Older `cpu_per_node` profiles retain DPDispatcher's tasks-per-node semantics;
migrate their CPU/task settings together. PBS keeps its existing resource
semantics and does not accept the Slurm-only fields. Memory is not injected:
omit memory flags to use site defaults, or set a site-valid request explicitly.
No partition, account, CUDA path or memory restriction is hardcoded.

`slurm_preflight` runs `sbatch --test-only` before launch. It can detect policy
rejection, but does not reserve capacity, prove GPU compatibility or guarantee
that a later submission is accepted. Its estimates may be misleading on
provider-modified Slurm installations.

Submission outcomes are distinct:

* `submitted`: a valid job ID was confirmed.
* `rejected`: preflight failed before launch, or the scheduler returned a nonzero
  status with a recognized policy/input rejection diagnostic.
* `uncertain`: a response was lost, a receipt was empty/invalid, or evidence
  contradicted itself. An empty temporary receipt is never success.

A nonzero scheduler-client exit alone is insufficient: communication errors
and unrecognized failures remain uncertain because the controller might have
accepted the job before its reply was lost.

The remote exit-status file is published last by rename. Exit status, stdout
and stderr are preserved in dispatcher state and local `submission-*` files.
Do not put credentials in scheduler arguments or resource flags; these
diagnostics are intentionally retained for troubleshooting.

For a rejection, correct the profile and run:

```bash
temper_bench job resubmit --bundle-write-root ./bundles \
  --resource-file corrected.yaml --dry-run
# Remove --dry-run after reviewing eligibility.
```

The Python equivalent is `manager.resubmit(replacement_resource=resource)`.
Optional bundle, resource-name, and domain filters narrow the selection. Without
filters, all matching bundles are considered once. This preserves the rejected attempt and diagnostics, cleans only its owned
remote directory, and creates a fresh attempt under the existing store lock.
For unknown outcomes, run `job recover` first. Slurm recovery checks the durable
receipt, then queue/accounting records by the unique attempt working directory.
No match is inconclusive and never authorizes resubmission. If accounting is
unavailable, verify the existing job's working directory yourself and pass
`--job-id`. PBS requires that explicit verification when its receipt is missing.
Never manually edit `job.json` to bypass uncertainty.

## Job-local caches and environment checks

Each family environment may set `cache_root: /tmp` (or another full path on the execution machine for temporary caches) and `required_executables: [python, dp]`. Activation runs first;
missing executables stop execution. The launch command is `bash run.sh`;
script and input errors are reported by the runtime. These are
installation checks, not GPU functional tests.
Set `check_dependencies: true` to run `python -m pip check` after activation.

Each job creates a unique owned cache child. Unset Triton, Inductor, Torch
extensions, CUDA and XDG cache variables point there. Existing values remain
unchanged. Successful jobs remove that owned child; failed jobs print and retain
its path for diagnosis. Site-managed job-private `/tmp` may disappear when the
allocation ends even after failure, so preserve needed logs in the attempt's
shared directory. Later jobs always create new scratch and never assume a prior
job's cache exists. User-specified cache directories are never deleted.

## Validation and upstream evidence

See [deployment validation records](deployment/) and the version-controlled
[deployment skill](../skills/install-temper-mlff/SKILL.md). PBS tests are mocked
scheduler/script tests unless a real PBS resource is explicitly recorded.

Upstream sources inspected on 2026-10-27:
[DPDispatcher releases](https://pypi.org/project/dpdispatcher/),
[MACE 0.3.16](https://pypi.org/project/mace-torch/0.3.16/),
[MatterSim 1.2.5](https://pypi.org/project/mattersim/1.2.5/),
[calorine 4.1](https://pypi.org/project/calorine/4.1/),
[GPUMD dump_xyz](https://gpumd.org/gpumd/input_parameters/dump_xyz.html).
The exact downloaded source URLs and hashes are in
[`compatibility/upstream-artifacts.json`](../compatibility/upstream-artifacts.json).
