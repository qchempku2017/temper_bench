---
name: install-temper-mlff
description: Deploy isolated TEMPER MLFF execution environments on an SSH-accessible Slurm or PBS cluster, validate fine-tuning and ASE inference in scheduled GPU jobs, and produce a tested TEMPER resource profile.
---

# Install TEMPER MLFF environments

Deploy the execution runtime required by the user's TEMPER checkout. Work through model families one at a time. Finish with reproducible environment records, actual GPU test results, a resource profile, and removal of deployment intermediates. Never describe an installed or importable package as a validated training environment.

## Establish the deployment contract

Read the selected TEMPER revision's `pyproject.toml`, `docs/mlff-bundles.md`, `docs/remote-jobs.md`, model catalog, spec builders, training adapters, calculators, and job-management resource schema. Record the commit and installed package version. Documentation may lag the implementation; inspect the commands that will actually run.

Discover the remote account, SSH endpoint and port, writable installation prefix, shared job directory, scheduler, GPU partitions, account/QoS requirements, driver, and available compiler/CUDA modules. Use a short scheduled GPU probe to inspect the GPU. Determine whether compute nodes have internet access. Do not assume that a provider's customized `sinfo` output is standard Slurm output or that its displayed zero available GPUs prevents a job from being scheduled.

For Slurm, preflight the generated scheduler script with `sbatch --test-only` before launching a new resource shape. Provider submission hooks may reject otherwise valid Slurm memory flags; inspect the actual default memory allocation. For current TEMPER, use resources.tasks_per_node=1 and resources.cpus_per_task=4 instead of cpu_per_node and duplicate custom flags. Legacy cpu_per_node retains its historical tasks-per-node meaning. Use slurm_preflight for optional preflight. Verify the effective allocation with `scontrol show job`; a site may allocate more CPUs than requested.

**Never run training, model inference, a model calculator, or a GPU smoke test on a login node.** Submit all such work as single-GPU scheduler jobs. Respect scheduler-provided `CUDA_VISIBLE_DEVICES`. Downloads, editing, environment installation, and compilation may run on login nodes only within the site's rules. Observe compiler parallelism limits.

Confirm the private key used by each transport. A provider may require a different key for file transfer than for an interactive login. Test an upload/download round trip and compare SHA256 hashes. Explicitly specify the tested private key. OpenSSH `scp` may use SFTP by default; record which protocol actually succeeded. Test TEMPER's Paramiko connection as well as OpenSSH.

Use verified host keys. In TEMPER, omitting `known_hosts` uses the default host-key store; it does not disable verification. Inspect the current implementation before relying on agent or automatic key discovery. Never copy private keys into deployment archives, bundles, or the cluster.

## Storage and downloads

Resolve symlinks in the chosen installation and job roots. Confirm quota and free space, including installation archives, extracted package caches, CUDA libraries, and final environments. Create a unique deployment staging directory and keep an ownership manifest of generated paths.

Distinguish connection stalls, download stalls, and shared-filesystem stalls. Bound SSH connection/command attempts and investigate a command that stops responding for two minutes. A large download or installation may legitimately take longer while making measurable progress. After a lost response, inspect the process and destination before retrying a mutation or submission.

On slow shared storage, keep temporary downloads, pip caches, and conda package caches on a site-permitted local scratch directory. Final environments and job inputs must be visible from compute nodes. Login-node `/tmp` is usually not shared. Do not reference it from a submitted job. Check temporary disk usage and remove only this deployment's files when finished.

Measure a small-file workload as well as bulk throughput: a filesystem can create directories quickly while each file close waits for remote upload. A temporary installation can be relocated with `conda-pack --dest-prefix FINAL_PREFIX`; that option rewrites embedded paths, while `--output` selects the archive destination. Copy into a new staging directory, verify completion, stop superseded installers, and only then replace an incomplete installation. Never merge a relocated tree into a prefix being modified by another installer. Compute-node scratch may be cleared or isolated between jobs, even when the same node is allocated again. Inspect Slurm namespace settings such as job_container/tmpfs and PrologFlags=Contain: job-private temporary directories are removed at termination. Do not expect a later job to reuse an earlier job's scratch or cache.

A valid tar archive can contain repeated paths. Standard extraction uses the last entry; a parallel extractor must deduplicate accordingly and must not race to create the same path. Log worker failures immediately and cancel queued work on failure; otherwise an executor can continue copying for hours while hiding the original exception. When resuming a partial extraction, verify existing bytes against the selected archive entry before reusing them.

Preserve file timestamps when extracting or copying environments. Invalidated `.pyc` files add source reads and may trigger thousands of bytecode writes on slow shared storage. Set `PYTHONDONTWRITEBYTECODE=1` for execution when appropriate. If restoring timestamps from an archive, exclude every source file edited after packing; do not make stale bytecode appear valid for changed code. Allow substantially more wall time for the first persistent-prefix run than for a local-scratch test.

Do not ignore `conda-pack` warnings about a cleared package cache. In conda-pack 0.9.2, unknown Unix binary files do not receive normal prefix replacement. Prefer retaining the installer cache when possible. For a prefix whose `conda-meta` still includes complete `paths_data`, the included `scripts/pack_offline_environment.py SOURCE_PREFIX FINAL_PREFIX OUTPUT.tar` restores the recorded relocation modes, checks that every recorded path maps to an installed file, and verifies that the old prefix is absent from those archive members. It requires conda-pack 0.9.2 and a final prefix no longer than the source prefix. It scans large unmanaged binaries in chunks and refuses unexpected embedded prefixes. Final GPU validation from the persistent prefix remains mandatory.

Download from upstream release channels. For inaccessible GitHub resources, use the user's workstation as a relay and transfer with the tested key. A reputable mirror is acceptable when the complete artifact is verified against a checksum published by upstream. Resume partial large downloads; do not repeatedly discard gigabytes after a wall-clock timeout. Verify ranged responses and file sizes when using segmented downloads.

Fetch **DeepMD DPA4 and DPA4C pretrained weights from AIS Square**. Consult their release descriptions and use portable `.pt` training checkpoints, not compiled `.pt2` exports. Pre-stage all models and runtime dependencies before submitting offline compute-node jobs. Do not silently substitute a different architecture or a randomly initialized model for a requested pretrained fine-tuning test.

## Conda and isolated environments

Install Miniconda below the user-approved persistent prefix and verify the official installer checksum. Initialize the user's shell as authorized, without activating the base environment automatically. Use that Miniconda installation for environment activation. In noninteractive scheduler scripts, source its `etc/profile.d/conda.sh` explicitly.

Create separate environments for MACE, MatterSim, SevenNet, and NEP. Install TEMPER itself in every execution environment. Install only its runtime dependencies; preprocessing and local submission extras are not generally needed on compute nodes. Select a Python version compatible with the resolved NumPy range and each backend; run pip check. Explicitly select a CUDA-enabled PyTorch build compatible with the actual driver and GPU architecture. Preserve CUDA visibility set by the scheduler.

For DeepMD, prefer the official self-contained GPU `.sh` release installer. Download every numbered part, verify each official digest, concatenate in numeric order, then execute the resulting script with a dedicated installation prefix. Activate its resulting prefix using the shared Miniconda `conda activate /absolute/deepmd/prefix`. **Do not replace this workflow with a conda-forge DeepMD installation.**

Select AIS Square checkpoint/explicit-JSON pairs compatible with the formal
release. Preserve the complete model section and checkpoint type-map order;
do not use presets. TEMPER reads a same-stem JSON beside each fine-tuning
checkpoint, or an explicitly supplied training_parameters.model mapping.
Do not upgrade to a development backend to rescue an incompatible checkpoint.
Record exact installer, checkpoint and JSON hashes. Test tiny fine-tuning,
freeze, ASE inference and temporary-export cleanup. If no candidate works,
reproduce with native DeepMD outside TEMPER, inspect the release code, search
open and closed upstream issues/PRs, and prepare a nonduplicate issue draft
with evidence and a proposed fix for review before posting.

The official installer may need Conda activation, but do not resolve DeepMD
from a Conda channel. Check its actual bundled dependencies and CUDA/compiler
requirements. Keep existing environments unchanged. Local WSL2 GPU validation
is permitted when explicitly requested; remote GPU validation always requires
a scheduler allocation. An import or pip check is not functional validation.

Use the versions required by the selected TEMPER revision, rather than treating this skill as a permanent version lock. The current integration targets are (consult recorded tests for acceptance):

| Family | Integration requirements |
| --- | --- |
| DPA4 / DPA4C | DeepMD-kit 3.2.0 official offline installer with verified explicit checkpoint/JSON pairs |
| MACE | mace-torch 0.3.16 |
| MatterSim | MatterSim 1.2.5 |
| SevenNet | sevenn 0.13.0 |
| NEP89 | TorchNEP 1.0.2, GPUMD 5.7, calorine 4.1 |

For MACE 0.3.16 with the referenced TEMPER revision, inspect the training-label keys: MACE defaults to `REF_*`, while TEMPER's standard ASE extxyz files use `energy`, `forces`, and `stress`. Explicitly configure the appropriate keys. Also check the local-checkpoint loading path before `E0s=estimated`: the pretrained model and newly constructed input tensors must use the same floating-point dtype. Preserve and document any necessary upstream or TEMPER patch; verify it in a GPU job rather than changing the scientific recipe merely to hide the error.

NEP training uses TorchNEP, not GPUMD's SNES training workflow. ASE inference uses calorine's `GPUNEP` and requires a working `gpumd` executable on PATH. Build the required GPUMD version for the target GPU, with suitable CUDA runtime libraries. Place the binary and any necessary reversible activation hooks in the NEP environment so a single activation suffices.

Use calorine 4.1 with GPUMD 5.7: its upstream dump_xyz support replaces the old deployment-specific calorine patch, and its NumPy metadata no longer imposes the conflicting ceiling. Current TEMPER explicitly inherits scheduler GPU visibility unless overridden. Read ../../docs/deployment-compatibility.md and ../../compatibility/upstream-artifacts.json before selecting versions. Build remaining MACE and MatterSim patches with ../../compatibility/build_backend.py; never edit installed packages. Preserve wheel/source/patch hashes and inspect existing environment changes before reusing deployment results.

Check `pip check`, package metadata, executable paths, and resolved environment dependencies. These are installation checks, not substitutes for GPU execution. Do not disable dependency checks or force incompatible packages merely to make installation finish.

For the audited formal 3.2.0 recipe, use the official CUDA 12.9 offline
installer and the August checkpoint/JSON pairs recorded in
`../../docs/deployment/pretrained-models.json`. Add `vesin-torch==0.6.1`
through normal dependency resolution: the installer alone lacks this DPA4
neighbor-list backend. Do not install a Linux NVIDIA driver inside WSL.
WSL export linking may require appending `/usr/lib/wsl/lib` to LIBRARY_PATH;
check that libcuda.so exists there first. This is a local environment setting,
not a TEMPER runtime default or a remote cluster CUDA path.

Use `../../compatibility/validate_deepmd_release.py` to separate native
checkpoint/backend failures from TEMPER failures. It requires a scheduler
allocation, except for explicitly requested WSL tests using `--local-wsl`.
It records training, freeze, ASE predictions, versions and cleanup in JSON.
Only after native validation should the complete TEMPER workflow be certified.

DeepMD freeze needs an unversioned `libcuda.so` at link time. The tested offline
installer does not include it. If the compiler reports `cannot find -lcuda`,
check the existing CUDA toolkit's `lib64/stubs/libcuda.so`, then append that
directory to `LIBRARY_PATH` through the site's resource configuration. Keep
existing search entries. Never put CUDA stubs on `LD_LIBRARY_PATH`, which is
used to load the real driver at execution time. WSL provides its driver link
library under `/usr/lib/wsl/lib`; verify that path before using it. This is an
environment correction, not a checkpoint or TEMPER source patch. Reuse a
completed trained checkpoint when only export needs retesting; a public
JobManager resubmission retries the whole pipeline and records a new attempt.

## Validate actual TEMPER workflows

For storage with slow individual file operations, an already relocated
conda-pack archive can be unpacked on local scratch, then copied with
`../../compatibility/copy_environment.py SOURCE NEW_PERSISTENT_PREFIX`.
The archive's --dest-prefix must match that destination. The tool refuses
existing destinations and copies independent files concurrently; it does not
rewrite paths or certify the environment. Never merge into a prefix still
being installed. Keep scratch and relocation in one login session, then run
installation checks at the persistent path before any scheduled GPU test.

Build tiny, physically sensible periodic toy datasets with energy, force, and stress labels. Use a few training frames, a separate validation frame, and a separate test frame. Include nonzero forces to avoid degenerate normalization. A toy-data smoke test verifies execution, not scientific accuracy or convergence.

Use RuntimeEnvironment.cache_root for owned per-job caches, preserving existing variables and removing owned caches after success. Retain failure diagnostics in shared storage because job-private scratch can disappear. Keep compilation caches on job-local scratch when the shared filesystem is slow: configure `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, and, when appropriate, `XDG_CACHE_HOME` before importing the backends. Setting only the Inductor directory does not relocate Triton's separate cache. DeepMD freeze may tune inference kernels even for a tiny dataset; inspect its log before interpreting a quiet period as a hang, and allow a bounded first-compilation budget.

The included `scripts/validate_mlff.py` implements this check for the referenced TEMPER API. From a fresh job scratch directory, after activating the intended environment, run `python /path/to/validate_mlff.py FAMILY /path/to/pretrained_model`. Supported families are `mace`, `mattersim`, `nep89`, `sevennet`, `dpa4`, and `dpa4c`. It requires a Slurm/PBS allocation and a working CUDA device, uses four Cu training frames plus separate validation/test frames, and writes `smoke-FAMILY/result.json`. MatterSim uses a two-frame batch. Run one family at a time; inspect and adapt the script if the user's TEMPER API or checkpoint element coverage differs.

For each family:

1. Select its smallest supported pretrained variant. For NEP, test the actual NEP89 release. Test DPA4 and DPA4C separately.
2. Build and package a real `MLFFTrainBundle` using the selected TEMPER API. Include dataset hashes when remote job management requires them. Set a minimal supported training budget and a small batch size.
3. Submit a single-GPU job that activates only the intended environment and runs `temper_bench run_pipeline` or the equivalent installed Python API. Bound wall time and CPU threads. Keep required model files local to the shared bundle; disallow implicit internet downloads in the job.
4. Confirm the requested final training step/epoch was reached and the expected fine-tuned artifact exists. Inspect the backend training log when execution fails.
5. Confirm the TEMPER ASE evaluation produced finite energies, forces, and stresses with correct shapes and units. Verify saved metadata identifies the intended packages and GPU.
6. For DeepMD, verify the actual native freeze command, inference using its exported model, and TEMPER's cleanup of the temporary export. DPA4 uses the PyTorch backend; DPA4C uses the exportable PyTorch backend and its graph-lowering options in the referenced TEMPER revision.
7. Record the scheduler job ID, state, exit code, GPU, model identity/hash, package versions, and observed assertions.

Where the installation claims general batch support, test a small multi-frame batch as well. MatterSim 1.2.5 has upstream reports of CPU/CUDA label-device mismatches and incorrect batched three-body indices. Batch size one does not repair a device mismatch. Inspect upstream code/issues, apply only a justified minimal patch, preserve its diff, and retest affected behavior. Report any remaining limitation. Do not silently change TEMPER's scientific defaults to bypass failures.

For a Windows submission host, test long paths as well as spaces in the account name. The old DPDispatcher 1.0.3 path joins a task path of `.` onto the Windows extended-length prefix and converts local upload paths to POSIX form; both can make existing files appear missing. Normalize local filesystem paths and preserve native separators, while retaining POSIX paths for the remote host. Current TEMPER owns its narrow transfer compatibility path; do not run the obsolete site-packages patch script. Verify the full round trip with scripts/validate_roundtrip.py.

When the user explicitly authorizes a local WSL2 GPU check, the validator also
accepts `--local-wsl`; it verifies the WSL kernel and still requires exactly
one visible GPU. This exception does not permit GPU work on cluster login
nodes. Keep local results distinct from scheduler reproduction.

When validating a checkpoint release range, enumerate the actual AIS Square
file lists before selecting models. A size range can include both materials
(OMat24) and molecular (OMol25) checkpoints. Do not infer that every checkpoint
with the same month is compatible from one successful size. Record the
matching JSON and checkpoint hashes for every tested pair. For the native
DeepMD validator in `../../compatibility/validate_deepmd_release.py`, use
`--molecular` for the OMol25 neutral-singlet test: it supplies charge/spin
values `[0, 1]` to training and ASE inference and uses isolated H2 geometries
in a large periodic cell. This is a functional test, not an accuracy result.
Keep native backend validation distinct from a full TEMPER JobManager test.

If Slurm accepts an array but leaves it pending, record its array/task IDs
and scheduler reason. Do not describe preflight start estimates as guarantees
or submit another copy after a connection interruption. Reconcile the receipt
and existing array first. An array throttle limits concurrent tasks; each
task must still request exactly one GPU.

After model tests, exercise at least one small job through TEMPER's actual remote submission path: upload, submit, monitor, collect predictions and final weights, and clean its owned remote attempt. Authentication or a manually submitted `sbatch` job alone does not certify the entire remote workflow.

## Deliver and clean up

Generate the resource YAML from the current `RemoteResource` schema. Include the real SSH endpoint/account/port, the tested local private-key path, a verified host-key store when necessary, a persistent shared `remote_root`, site-valid single-GPU scheduler resources, and an `environments` entry for every supported TEMPER family. Each entry should identify the common `conda_init` and the tested environment name or absolute prefix. DPA4 and DPA4C can share one compatible DeepMD prefix.

Validate the YAML with TEMPER. Do not claim an environment in the profile is ready if its required GPU test failed or was not run. Distinguish submission-host paths from remote paths; Windows and WSL need different local key paths.

Save a concise validation record and reproducibility information before cleanup. Remove only owned installers, archive parts, toy datasets, downloaded test weights, build trees, temporary caches, and logs that are no longer required. Do not remove installed packages, activation hooks, user files, or a shared cache used by other work. Verify no deployment job or installer is still using a path before deleting it. Never clean by broad patterns across a user's entire run directory.

An explicit user request to remove obsolete installations overrides that
default preservation rule. Resolve each removal target to its absolute path,
check active jobs and processes, preserve package/patch provenance, and check
that the retained installation does not depend on the target. Remove stale
Conda registrations and update activation references when applicable. If the
request could include unrelated working backend environments, clarify that
scope before deleting those environments.

If external storage, network access, scheduler capacity, or an upstream defect prevents completion, report precisely what is installed, what was actually tested, which processes/jobs remain active, and what evidence blocks the next step. Preserve only the diagnostic material necessary to resume; do not label partial deployment as successful.

## Primary references

- [TEMPER repository](https://github.com/qchempku2017/temper_bench)
- [Miniconda installers and SHA256](https://repo.anaconda.com/miniconda/)
- [DeepMD offline installation](https://docs.deepmodeling.com/projects/deepmd/en/latest/getting-started/install.html)
- [DeepMD development installation](https://docs.deepmodeling.com/projects/deepmd/en/latest/install/easy-install-dev.html)
- [mpi4py binary installation and MPI compatibility](https://mpi4py.readthedocs.io/en/latest/install.html)
- [Slurm job-private temporary namespaces](https://slurm.schedmd.com/namespace.html)
- [Slurm MPI launch modes](https://slurm.schedmd.com/mpi_guide.html)
- [AIS Square DPA4](https://www.aissquare.com/models/detail?pageType=models&id=423&name=DPA4-OMat24)
- [AIS Square DPA4C](https://www.aissquare.com/models/detail?pageType=models&id=433&name=DPA4C-OMat24)
- [MACE installation](https://mace-docs.readthedocs.io/en/latest/guide/installation.html)
- [MatterSim upstream issue 163](https://github.com/microsoft/mattersim/issues/163)
- [SevenNet documentation](https://sevennet.readthedocs.io/en/latest/)
- [TorchNEP](https://github.com/mushroomfire/torchnep)
- [GPUMD](https://github.com/brucefan1983/GPUMD)
