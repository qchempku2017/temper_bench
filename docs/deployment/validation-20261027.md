# Compatibility validation — 2026-10-27

> Historical initial implementation report. For the subsequent review fixes,
> completed GPU jobs, formal-release DeepMD pairs and current cleanup status,
> see [review-validation-20261027.md](review-validation-20261027.md).

**Status: implementation and local regression validation complete; GPU acceptance
is pending scheduler capacity. Do not treat this as a completed GPU deployment.**

The initial branch was clean at
`828769a3400b054120c2448bfc3ea7017034669e`. Changes are in the working tree, not a
new commit. The queued tests use a source snapshot based on that revision plus
these implementation changes; its archive SHA256 is
`6484c6cecf2dee23028d0d2952d68eecec71bb81a7f91f56531fc57840a5c33b`.
Subsequent edits clarify docstrings and conservatively classify unrecognized
scheduler-client errors as uncertain; these affect the submission host, not the
queued model calculations. The GPU harness records every
loaded TEMPER source-file hash as well as the archive hash, package inventory,
checkpoint hash and dependency source hashes. Existing installed TEMPER copies
are bypassed by an explicit source `PYTHONPATH` for these tests.

## Issue-to-fix mapping

| Issue | Fix | Validation and limits |
| --- | --- | --- |
| Windows paths | Native normalized archive/upload/download paths in TEMPER's SSH context; remote paths remain POSIX | Real Windows long-path, spaces, dot-component and extended-prefix IO regressions pass. Actual MACE upload and submit passed from an extended path containing spaces. Download of real model results awaits execution. |
| DeepMD schema | Top-level optimizer; family-specific model types; idempotent legacy normalization with conflict errors | Configuration/legacy/conflict regressions pass. Freeze, inference and export cleanup are queued for both families. |
| MACE labels/dtype | Explicit ASE keys for new and old bundles; verified dtype patch in a versioned wheel recipe | Configuration and override regressions pass; MACE wheel builds. GPU round trip queued. |
| MatterSim device/batching | Version-controlled backend patch and wheel builder; no version-only defect warning | Source hashes match freshly patched upstream sources; Linux wheel builds. Two-frame training batch queued. |
| NEP/NumPy/GPUMD | Upstream calorine 4.1; preserve scheduler GPU visibility; relax TEMPER's unnecessary exact NumPy pin | Isolated calorine installation and pip check pass. No calorine source changes in selected 4.1 environment. GPUMD binary provenance retained. GPU fine-tuning/inference queued. |
| Slurm semantics | Explicit tasks_per_node/cpus_per_task; optional test-only preflight; no injected memory flag | Generated header and rejection tests pass; real MACE request is 1 node, 1 task, 4 CPUs, 1 GPU, with site-default 40 GB. |
| Submission recovery | Durable status/stdout/stderr; rejected versus uncertain states; locked public retry; scheduler reconciliation | Rejection, interrupted response, empty/valid receipts, explicit retries and scheduler reconciliation tests pass. Actual MACE persisted-state reload/monitoring passed. |
| Runtime caches | Per-job owned caches, respect existing variables, success cleanup/failure retention; optional executable/pip checks | Actual Bash success/failure cleanup checks pass. No site CUDA path is embedded in TEMPER. |
| PBS | Existing resource and status semantics preserved | Mocked header/status and receipt coverage only; no real PBS account was available. |

## Local tests

Windows configured interpreter: Python 3.12.10. DPDispatcher 1.0.3 was not edited.

* Initial targeted suite after fixes: 168 passed, 2 skipped.
* Full repository suite with NumPy 2.5.2: 440 passed, 3 skipped.
* Full repository suite with isolated NumPy 2.3.5: 437 passed, 3 skipped
  (before three additional scheduler/PBS tests); the later targeted run passed
  77 tests with one skip.
* Final recovery/CLI/dependency-check boundary run: 87 passed, 1 skipped.
* Final focused compatibility run, including actual Bash cache cleanup and
  native Windows downloads: 18 passed.
* Final conservative scheduler-client classification and JobManager run:
  83 passed, 1 skipped (including the added ambiguous-client-error cases).
* `pip check` passed for the resolved local NumPy 2.3.5 setup and all six remote
  checkpoint-specific installation checks. The local editable metadata was
  refreshed after removing the exact NumPy pin; dependency conflicts were not
  ignored.
* Deployment skill structural validation passed. MACE and Linux MatterSim
  patched-wheel build provenance is retained beside this file.

The skipped tests are platform/optional-runtime tests, not successful GPU or
real PBS tests. NumPy 2.5.2 emits existing ASE deprecation warnings.

## Pending single-GPU jobs

At the last scheduler check on 2026-10-27, all jobs were `PENDING (Priority)`.
The scheduler estimated 2026-10-29 09:25:42 for the first job. Estimates are not
guarantees. `hp_4090` requires account `hp4090`; this account's preflight was
rejected. No 5090 or multi-GPU execution was attempted.

| Family | Actual job ID | Environment |
| --- | --- | --- |
| MatterSim | 1742129 | Existing mattersim prefix, audited two-file patch |
| SevenNet | 1742131 | Existing sevennet prefix |
| DPA4 | 1742133 | Existing dedicated deepmd-kit prefix |
| DPA4C | 1742135 | Same dedicated deepmd-kit prefix |
| MACE | 1742138 | Existing mace prefix, through native-Windows JobManager |
| NEP89 | 1742154 | Isolated nep-venv with calorine 4.1 over the existing NEP prefix |

Each requests one RTX 4090, one task, four CPUs, site-default memory and a
one-hour limit. Test-only probes are not included as computational jobs.
No training, model calculator or GPU functional test ran on a login node.

The six earlier successful jobs from the supplied deployment report
(1740884, 1740940, 1741013, 1741101, 1741201, 1741207) tested an older revision
and patched installations. They are historical evidence, **not acceptance
results for this change**.

## Environment audit and provenance

`audit-*.json` lists exact installed distributions and changes against installed
wheel RECORD hashes. Those hashes are an audit aid, not a cryptographic signature.
The selected calorine 4.1 overlay has no modified Python files. The legacy
calorine 3.5+temper.1 installation remains unchanged and is recorded separately.
The MACE environment also retains the old label-default patch in arg_parser.py;
TEMPER now supplies keys explicitly, so new installations do not need that patch.
Old installed TEMPER DeepMD/MACE patches are bypassed by the test source snapshot.
Their diffs are retained in `historical-patches/` solely to explain the audits;
do not apply them to new installations.

`installation-*.json` records successful checkpoint hashes, required backend
versions and compatibility source hashes. It explicitly says that no GPU
functional test was performed by the checker. DPA4 and DPA4C were downloaded
from AIS Square and match the recorded hashes. The NEP GPUMD executable targets
sm_89, as recorded in `gpumd-component.json`.

## Resume and cleanup

No choice to cancel the pending tests has been received. They remain queued
with their inputs intact. Existing working environments were preserved.
Completed build environments, downloaded dependency source trees/archives,
temporary NumPy/build-tool installations and routine raw installation/test
logs were removed after preserving their concise records. Pending jobs'
models, toy data, selected NEP overlay, source snapshot and local job history
are intentionally retained; final cleanup awaits their terminal results.

Owned remote staging root:
`/data/run01/scxm254/temper-compat-20261027`.
MACE's separate owned JobManager attempt:
`/data/run01/scxm254/temper_jobs/temper_0af4ec0089fe4ceab37e1f0678e6e6bd`.
Local resumable inputs/history are in `.compat-work/`, excluded from Git.

To monitor and complete MACE collection, run from this checkout:

```powershell
.venv/Scripts/python.exe skills/install-temper-mlff/scripts/validate_roundtrip.py check `
  '.compat-work/round trip with spaces' .compat-work/models/mace .compat-work/resource.yaml
```

This submits nothing. On success it verifies finite output shapes, preserves
JobManager's downloaded hashes and cleans only the owned remote attempt.
Copy the resulting roundtrip-result.json into this evidence directory before
removing local models/data/results.

For the other five jobs, retrieve `smoke-FAMILY/result.json` from the remote
staging root, confirm Slurm state/exit status and the one-GPU allocation, and
preserve those concise records. Inspect failure logs before retrying anything.
The DeepMD records must confirm freeze and temporary export cleanup. Only after
all jobs are terminal and evidence is retained may the owned staging root and
remaining local models, toy data and raw logs be removed. Never delete queued
or running jobs' inputs, existing environments, or shared user caches.
