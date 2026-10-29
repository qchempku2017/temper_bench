# Review fixes and validation — 2026-10-27

The reviewed source changes pass the Windows regression suite: **452 passed,
4 skipped**. Native DeepMD 3.2.0 fine-tuning, freeze, ASE inference and export
cleanup pass on the local RTX 4060 Ti through WSL2 for both selected August
checkpoints. Both reviewed TEMPER pipelines also pass end to end in WSL with
the final Vesin 0.6.1 / vesin-torch 0.6.1 recipe, including legacy configuration
normalization and temporary export cleanup. Their
[DPA4](wsl-temper-dpa4.json) and [DPA4C](wsl-temper-dpa4c.json) loaded Python
source hashes are preserved alongside the final reviewed wheel's manifest.
Remote reproduction also passes: DPA4 training in job **1742856** followed by
export/inference recovery in **1742900**, and the complete DPA4C JobManager
round trip in **1742894**. All computational jobs used one GPU. The initial
linker failures and environment correction are preserved below.

The work is uncommitted on base revision
`828769a3400b054120c2448bfc3ea7017034669e`. The exact reviewed source files and
wheel are identified in [the source manifest](review-source-manifest.json).
The wheel SHA256 is
`a62756210e1342d7087f88967285070e89c0e94c55bd7d5c29302c0d241bda0f`.
Earlier local pipeline tests used wheel
`8867955e02c85c4aa08892a46cfc709adf001807aca9964f5a85330ba4e12225`.
The final wheel adds API annotations and docstrings in five files and corrects
catalog dependency notes. Training and inference algorithms are unchanged.
The manifest retains both sets of source hashes and lists the changed files.
The final regression rerun passed 452 tests with four skips in 33.29 seconds.

## Responses implemented

| Review comment | Result | Validation |
| --- | --- | --- |
| 1. Installation extras | Workstation installs `remote,preprocess`; add `gpu` for local GPU entropy and `bohrium` only when used. Execution environments need base TEMPER plus their backend. | README, project metadata and deployment instructions agree. |
| 2. DeepMD release/checkpoint selection | Prefer the official 3.2.0 GPU offline installer. Add tested August checkpoint/JSON pairs and remove September catalog entries requiring a development backend. | Both selected pairs pass WSL and Paratera training, freeze, ASE and export cleanup. |
| 3. Explicit configuration | Embed the matching JSON's model section, including its ordered full `type_map`; reject presets. Keep `--use-pretrain-script`, which copies descriptor/fitting settings rather than every outer model field. | Configuration generation, explicit overrides, element order, aliases, legacy optimizer normalization and conflicts have regression coverage. |
| 4–5. MACE label configuration | Set `energy`, `forces`, `stress` defaults when building a specification; preserve overrides. Remove duplicated runtime defaults and old-YAML migration. | New-spec/override tests pass. Regenerate old MACE bundles. The separately built dtype fix remains necessary for the audited upstream version. |
| 6. Python formatting | Wrap source and tests to 79 columns and preserve literal command/configuration semantics. | Full suite passes after formatting. |
| 7–8. Failed-submission recovery | One `resubmit` API supports confirmed rejections and failed calculations, filtering and dry runs. Remove `retry_submission`. Preserve history, locking and cleanup errors; exclude unknown outcomes and stale attempts. Local recovery also rejects empty or malformed acknowledgments. | Bulk selection, cleanup failure/audit history, later recovery, scheduler rejection, receipt ambiguity and CLI regressions pass. |
| 9. Launch command | Run `bash run.sh` after environment setup without redundant file-readability probes. Bash returns launch failures; its parent retains the cache cleanup trap. | Local and remote command-generation tests pass. |
| 10. Explanations | Expand caller-facing documentation of resources, paths, lifecycle states, environment activation and recovery. | Docstrings use NumPy sections for the changed APIs; documentation describes defaults, side effects and failure conditions. |
| 11. Preflight naming | Use `slurm_preflight`. Continue accepting `slurm_test_only` when reading old profiles. | Alias loading and new-name serialization tests pass. Preflight does not reserve a GPU. |
| 12. Custom flags | Validate entries and then work with a typed `list[str]`. Reject non-list or non-string inputs and duplicate task/CPU requests. | Invalid-value, Slurm-header and PBS regression tests pass. |

PBS coverage is mocked scheduler and script coverage. No real PBS resource was
available. The four skipped tests are optional/platform coverage, not GPU
successes. ASE emits NumPy 2.5 deprecation warnings in the Windows suite.

The native Windows submission environment uses Python 3.12.10,
DPDispatcher 1.0.3 and Paramiko 5.0.0. Every installed DPDispatcher Python
file matches its wheel's SHA256 RECORD entry; no deployment edits remain in
those files. The [submission environment record](windows-submission-environment.json)
contains the versions and source hashes. Windows compatibility is provided by
TEMPER's scoped subclass, without modifying installed dependency files.

## Reproduce the formal DeepMD environment

Use the [official v3.2.0 release assets](https://github.com/deepmodeling/deepmd-kit/releases/tag/v3.2.0)
and the [offline installation instructions](https://docs.deepmodeling.com/projects/deepmd/en/latest/getting-started/install.html).
Download the four `deepmd-kit-3.2.0-cuda129-Linux-x86_64.sh.00` through `.03`
parts. Their URLs, lengths and individual SHA256 values are recorded in
[deepmd-release-validation.json](deepmd-release-validation.json). Verify each
part before concatenating, then verify the complete installer:

```bash
cat deepmd-kit-3.2.0-cuda129-Linux-x86_64.sh.0{0,1,2,3} > deepmd-installer.sh
sha256sum deepmd-installer.sh
# Expected: a702a39eb408617326b3804582b1e1c6292981deed47ccdad9adc8ad7971fb57
bash deepmd-installer.sh -b -p /YOUR/PERSISTENT/PREFIX/deepmd-3.2.0
source /YOUR/PERSISTENT/PREFIX/deepmd-3.2.0/etc/profile.d/conda.sh
conda activate /YOUR/PERSISTENT/PREFIX/deepmd-3.2.0
python -m pip install -c compatibility/constraints/deepmd.txt \
  TEMPER_WHEEL 'vesin-torch==0.6.1' 'dpdata==1.1.0'
python -m pip check
```

Replace the prefix with a new persistent directory and `TEMPER_WHEEL` with the
reviewed wheel's path. This uses the vendor's bundled installation, not a
Conda-channel DeepMD build. Conda activation selects its Python and libraries;
it does not download a different DeepMD package. On a disconnected cluster,
relay compatible Linux wheels and add `--no-index --find-links WHEELHOUSE` to
the installation command. Keep normal dependency resolution enabled.

On Paratera, direct package extraction into shared storage was too slow. The
same verified installer was therefore run in a new login-node `/tmp` directory,
then relocated with `conda-pack --dest-prefix FINAL_PREFIX`. This option rewrites
environment paths for the permanent destination; merely copying a Conda
directory can leave executables pointing at the temporary installation. The
archive is extracted into a new persistent prefix, followed by activation,
`pip check` and checkpoint/JSON checks at that final location. No GPU test uses
the login-node scratch path. Scratch installation, packing and relocation run
in one SSH session because another session or compute job may see a different
`/tmp`. The incomplete direct-install prefix is retained only until the
replacement is verified, then removed as an owned installation intermediate.

Serial extraction to this site's shared filesystem was also slow. For that
case, unpack the relocated archive on local scratch and use the optional
[`copy_environment.py`](../../compatibility/copy_environment.py) installation
tool to copy independent files concurrently into a **new** persistent prefix:

```bash
mkdir /LOCAL_SCRATCH/relocated
tar -xzf environment.tar.gz -C /LOCAL_SCRATCH/relocated
python compatibility/copy_environment.py /LOCAL_SCRATCH/relocated \
  /YOUR/PERSISTENT/PREFIX/deepmd-3.2.0 --workers 32
```

The archive must already encode that exact permanent path through
`conda-pack --dest-prefix`. Concurrent copying reduces time spent waiting for
individual file operations; it does not modify package code or replace
installation checks. The tool preserves symbolic links and refuses an existing
destination. Adjust concurrency to site policy. Windows unit coverage and a
POSIX check verified byte copying, empty directories, link preservation and
overwrite refusal; backend acceptance still requires the scheduled tests.

The tested installer contains DeepMD 3.2.0, Python 3.12.14, PyTorch 2.12.0 with
CUDA 12.9, NumPy 2.5.2 and Triton 3.7.0. Added packages include ASE 3.29.0,
dpdata 1.1.0, bundled Vesin 0.6.1 and added vesin-torch 0.6.1. Complete inventories are in the
JSON evidence. **No DeepMD or PyTorch installed source was patched.**

DPA4's first attempt failed because the installer did not include a usable
neighbor-list implementation. Installing the supported `vesin[torch]` extra
resolved that dependency; no model setting was changed to hide the error.
On WSL only, the first DPA4C export failed to link `libcuda`. After checking
`/usr/lib/wsl/lib/libcuda.so`, the test appended `/usr/lib/wsl/lib` to
`LIBRARY_PATH`. The completed training was reused, and freeze/inference passed.
Do not install a Linux NVIDIA driver inside WSL or copy this WSL-specific path
into a cluster resource profile.

## Checkpoint pairs and GPU results

| Family | AIS Square checkpoint | Matching JSON | Local result |
| --- | --- | --- | --- |
| DPA4 | `DPA4-Nano-OMat24-v20260805.pt` | Same stem, `.json` | RTX 4060 Ti 8 GB: training, freeze, ASE, cleanup passed |
| DPA4C | `DPA4C-Air-OMat24-v20260819.pt` | Same stem, `.json` | RTX 4060 Ti 8 GB: training, freeze, ASE, cleanup passed |

Download URLs and checkpoint/configuration hashes are in
[pretrained-models.json](pretrained-models.json). The full explicit model
sections were retained, including their 118-element order. Each native test
used two four-atom Cu training frames, one validation frame and one test frame,
with nonzero forces and a one-epoch budget. ASE returned finite energy in eV,
forces with shape `(4, 3)` in eV/angstrom, and stress with shape `(3, 3)` in
eV/angstrom³. Temporary exports and successful-run caches were removed.
These are workflow checks, not accuracy or convergence benchmarks.

The first exploratory runs used Vesin/vesin-torch 0.6.2. They passed GPU
execution, but `conda-pack` later detected that pip had overwritten files owned
by the installer's `python-vesin` 0.6.1 package. Those results remain recorded
with their actual versions; that installation recipe is superseded. The final
recipe retains the bundled Vesin and adds only the
[matching vesin-torch 0.6.1 wheel](https://pypi.org/project/vesin-torch/0.6.1/).
The vendor package was restored from the verified installer's own archive in
the exploratory environments, then dependency and relocation checks were rerun.
No Conda channel replacement or ignored consistency check is required for a
fresh installation using the final recipe.

Run the reusable native validator inside a scheduled single-GPU allocation:

```bash
python compatibility/validate_deepmd_release.py dpa4 CHECKPOINT MATCHING_JSON OUTPUT
```

Use `dpa4c` for that family. Paths resolve from the working directory. Output
must be new so earlier evidence is preserved. For the explicitly authorized
local WSL test, add `--local-wsl`; this does not permit a cluster login-node
test. `--resume-export` can reuse successful training after an export failure,
but only with the same checkpoint and configuration hashes.

The September model cards require a post-October-1 development commit despite
also mentioning “3.2.0”. They have been removed from the supported catalog.
Other August sizes and MatPES entries retain metadata but have not gained GPU
certification from these two tests. Since compatible formal-release pairs were
found, the requested fallback GitHub issue claiming that no AIS checkpoint
works is not warranted and was not posted.

## Remote evidence and cleanup

Remote tests use the verified ECDSA key, existing known-host verification,
one task, four requested CPUs and exactly one GPU, with no explicit memory
request. The new installation is isolated under
`/data/run01/scxm254/temper-review-20261027/deepmd-3.2.0`; existing backend
environments remain unchanged. The installer reports `sm_120` support in its
PyTorch build, permitting a scheduled RTX 5090 compatibility test. GPUMD is
not part of those DeepMD tests and is not thereby certified for RTX 5090.

Earlier reviewed-task jobs **1742129 (MatterSim)** and **1742131 (SevenNet)**
completed successfully on RTX 4090. MatterSim used a two-frame training batch.
Their exact source hashes, dependency patches, versions and results are kept
in [completed-earlier-gpu-tests.json](completed-earlier-gpu-tests.json).
They used the earlier source snapshot, not the wheel identified above.
The earlier development-backend checkpoint provenance is retained in
[pretrained-models-historical.json](pretrained-models-historical.json), and the
initial job list remains in [the historical report](validation-20261027.md).

All six earlier jobs subsequently completed with scheduler exit code `0:0`:

| Family | Job ID | Observed result |
| --- | --- | --- |
| MatterSim | 1742129 | Two-frame training batch and ASE passed |
| SevenNet | 1742131 | Training and ASE passed |
| DPA4, development backend | 1742133 | Training, freeze, ASE and export cleanup passed |
| DPA4C, development backend | 1742135 | Training, freeze, ASE and export cleanup passed |
| MACE | 1742138 | Complete native-Windows JobManager round trip passed |
| NEP89 | 1742154 | TorchNEP training and calorine 4.1 / GPUMD 5.7 ASE inference passed |

The [MACE receipt and verified hashes](mace-roundtrip-1742138.json) and
[audit events](mace-roundtrip-events.jsonl) preserve its upload, submit, monitor,
download and owned-remote-attempt cleanup history. Its calculation used the
earlier source snapshot; collection used the reviewed submission code. Earlier
manual test datasets, downloaded models and raw logs have been removed after
saving these records. The installed NEP environment and source provenance remain.

Formal-release execution and the complete reviewed-wheel JobManager round trip
are verified. The [final scheduler records](formal-final-scheduler.json) show
exit code `0:0` for **1742900** and **1742894**, each with one node, four CPUs
and one GPU in `gpu_5090`. The site supplied 63000 MB of memory through its
default; no explicit memory directive was submitted. DPA4 directly reported
an RTX 5090. This does not extend GPUMD's separate RTX 4090 validation.

The [DPA4 evidence](formal-dpa4-1742900.json) verifies the unchanged trained
checkpoint, freeze, finite ASE arrays, export cleanup and loaded TEMPER source
hashes. The [DPA4C receipt and hashes](dpa4c-roundtrip-1742894.json),
[audit history](dpa4c-roundtrip-events-1742894.jsonl),
[prediction metadata](dpa4c-predictions-1742894.json) and
[export cleanup check](dpa4c-export-cleanup-1742894.json) establish the complete
native-Windows upload/submit/monitor/download/verify/cleanup cycle. Its native
Windows workspace contained spaces and the extended-length prefix.
The [installed source audit](formal-deepmd-environment.json) found no modified
or unverifiable Python files in the audited DeepMD and TEMPER distributions.

The first formal-release jobs, **1742856 (DPA4)** and **1742859 (DPA4C)**,
completed training on RTX 5090 but failed during freeze with
`x86_64-conda-linux-gnu-ld: cannot find -lcuda`. The offline installer lacks
the unversioned driver link library needed by AOTInductor. The existing site
toolkit supplies `/data/apps/cuda/12.8/lib64/stubs/libcuda.so`; adding its
directory to `LIBRARY_PATH` corrects linking without changing model settings
or installed source. Stubs must never be added to `LD_LIBRARY_PATH`, because
runtime loading needs the real driver. This site path appears only in the
Paratera resource example and test records.

The [initial scheduler records](formal-first-scheduler.json) preserve both
failures. DPA4's trained checkpoint is retained for export-only recovery.
For DPA4C, public `synchronize()` collected and cleaned the failed attempt;
`resubmit(dry_run=True)` then reported eligibility, and
`resubmit(replacement_resource=corrected_profile)` created attempt 002,
job **1742894**. No internal state was edited. A prior retry while cleanup
was deliberately deferred correctly returned a skipped outcome.

The corrected DPA4 export-only job **1742900** passed without repeating
training. The corrected DPA4C job **1742894** passed the full pipeline; public
resubmission creates a new calculation attempt and therefore repeats its tiny
training epoch. The failed attempt and its cleanup remain in the
[first-attempt history](dpa4c-roundtrip-events-1742859.jsonl). The
[failure record](formal-initial-failures.json) includes the precise linker
error, link-library hash and cache policy. A CPU-only link check confirmed
that the resulting library requests the real `libcuda.so.1` at runtime; the
probe was never loaded or executed on the login node.

The high-priority partition rejected the verified account; the
[eligibility record](high-priority-eligibility.json) preserves that result.
Both recovery jobs waited for normal queue priority. Preflight supplied
unrealistic far-future start estimates, reinforcing that it is a request
validation check, not a scheduling forecast.

Task-owned downloaded models, toy datasets, installers, relocation archives,
incomplete environments and raw logs are removed after saving this evidence.
The persistent WSL and Paratera installations are retained. Successful jobs
removed their owned caches. Failed jobs used job-private temporary storage;
its visibility and lifetime after allocation are scheduler-controlled, and
no subsequent job relied on it. See [the cleanup record](cleanup-review.json)
for removed paths, retained installations and that limitation.
