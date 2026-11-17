# Backend deployment

Start with the [deployment skill](../../skills/install-temper-mlff/SKILL.md).
It describes host discovery, installation, scheduled GPU validation, the
TEMPER submission round trip, and cleanup. Use this page for the recorded
backend recipes and the tools that support those steps.

Run commands below from the repository root with the intended environment
activated. Keep the checkout available: the tools read its patches, constraints,
and checkpoint catalog.

## Deployment inputs

- [Checkpoint catalog](pretrained-models.json): source URLs and SHA256 hashes,
  including matching DeepMD JSON files and backend requirements.
- [Slurm resource template](slurm-resource.example.yaml): replace the endpoint,
  credentials' local paths, shared storage, GPU partition, and environment prefixes.
  Validate it against the current resource schema and the target cluster.
- [Backend tools](../../tools/backend_setup/): installation and validation
  scripts, dependency constraints, and the two required upstream patches.
- [Remote job guide](../remote-jobs.md): resource fields, submission, recovery
  after uncertain outcomes, and cache behavior.

## Recorded recipes

These are the repository's tested version combinations, not a claim about the
latest upstream releases. A different checkpoint or dependency version needs
its own validation.

| Family | Recorded dependencies | Required setup |
| --- | --- | --- |
| MACE | mace-torch 0.3.16, e3nn 0.4.4 | Build the local-checkpoint dtype patch |
| MatterSim | MatterSim 1.2.5 | Build the device-transfer and three-body-index patch |
| NEP89 | TorchNEP 1.0.2, GPUMD 5.7, calorine 4.1 | Build GPUMD for the target GPU; use upstream calorine |
| SevenNet | sevenn 0.13.0 | No additional source patch |
| DPA4 / DPA4C | Official DeepMD 3.2.0 CUDA 12.9 offline installer, PyTorch 2.12.0, vesin and vesin-torch 0.6.1 | Use a cataloged checkpoint and matching explicit JSON |

Install TEMPER in each execution environment. Use separate environments for
independent backends; DPA4 and DPA4C can share the recorded DeepMD environment.
Select CUDA PyTorch for the actual driver and GPU. The Linux Python 3.12
[constraints](../../tools/backend_setup/constraints/) describe recorded
environments; DeepMD uses its official offline installer instead of those
constraints as an installation recipe.

For patched MACE and MatterSim wheels:

```bash
# In an isolated build environment:
python -m pip install build
python tools/backend_setup/build_backend.py mace-torch wheelhouse
python tools/backend_setup/build_backend.py mattersim wheelhouse

# Install only the matching wheel in each backend's environment:
python -m pip install -c tools/backend_setup/constraints/mace.txt \
  TEMPER_WHEEL wheelhouse/mace_torch-0.3.16+temper.1-*.whl
python -m pip check
```

The builder verifies upstream archives against
[recorded hashes](../../tools/backend_setup/upstream-artifacts.json), applies
the patch, and writes wheel/source/patch hashes beside the wheel. It also
accepts `--archive PATH` for a previously downloaded archive. Build MatterSim
on the target Linux/Python architecture and install with its matching constraints.
Do not edit installed packages or bypass dependency resolution.

MACE's patch aligns the local checkpoint dtype before energy-offset estimation.
TEMPER supplies ASE label keys explicitly. MatterSim's patch moves batches and
labels to the model device and fixes three-body edge offsets; validate it with
a two-frame training batch.

## DeepMD requirements

Download the official GPU offline installer and verify its published hashes.
Use AIS Square training checkpoints with a same-stem explicit JSON, preserving
the complete model section and element order. Add vesin-torch 0.6.1 through
normal dependency resolution. Do not switch to a development backend to make
an incompatible checkpoint load.

Freeze requires an unversioned `libcuda.so` at link time. If linking fails,
verify the site's CUDA toolkit stub and append its directory to `LIBRARY_PATH`.
Keep it out of `LD_LIBRARY_PATH`, which must load the real driver.
On WSL, check `/usr/lib/wsl/lib` instead. These paths belong in the site's
resource profile, not in TEMPER defaults.

## Verification tools

| Tool | Purpose |
| --- | --- |
| `tools/backend_setup/check_installation.py FAMILY CHECKPOINT OUTPUT.json` | Check a cataloged checkpoint, its configuration, package versions, patch hashes, executables, and `pip check` |
| `tools/backend_setup/audit_environment.py OUTPUT.json` | Record installed versions, backend source changes, and the GPUMD binary hash without importing models |
| `tools/backend_setup/validate_deepmd_release.py FAMILY CHECKPOINT CONFIGURATION OUTPUT_DIR` | Test native DeepMD training, freeze, ASE predictions, and export cleanup |
| `skills/install-temper-mlff/scripts/validate_mlff.py FAMILY CHECKPOINT` | Test TEMPER training and evaluation inside a GPU allocation |
| `skills/install-temper-mlff/scripts/validate_roundtrip.py` | Test remote submission, collection, and cleanup; inspect its CLI arguments before running |

Run Python scripts with the target environment's interpreter. The installation
checker accepts only recorded checkpoint/configuration hashes; a new model needs
its own established requirements. Installation checks alone do not establish
successful training or inference. Run GPU work through the scheduler.
The validators' `--local-wsl` option is only for explicitly authorized local WSL
checks. Preserve finite energy/force/stress checks, array shapes, scheduler
receipts, environment details, and source hashes with each new validation.

For relocation on slow shared storage, use
`skills/install-temper-mlff/scripts/pack_offline_environment.py` as described in
the skill. `tools/backend_setup/copy_environment.py SOURCE NEW_DESTINATION`
copies an already relocated environment; it does not rewrite embedded paths.
The archive's destination prefix must match the final location. Validate from
that persistent location before using it in a resource profile.
