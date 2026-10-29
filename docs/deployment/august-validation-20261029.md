# August checkpoint compatibility with formal DeepMD-kit 3.2.0

This validation covers the 16 AIS Square August checkpoints from Nano
through Air. It excludes Plus and larger models. The exact source URLs,
file sizes and SHA256 hashes are in
[`august-checkpoint-inputs-20261029.json`](august-checkpoint-inputs-20261029.json).
The public file lists were inspected on 29 October 2026:

| Family and training dataset | Release | Sizes |
| --- | --- | --- |
| [DPA4 OMat24](https://www.aissquare.com/models/detail?pageType=models&id=423&name=DPA4-OMat24) | 20260805 | Nano, Mini, Neo, Air |
| [DPA4C OMat24](https://www.aissquare.com/models/detail?pageType=models&id=433&name=DPA4C-OMat24) | 20260819 | Nano, Mini, Neo, Air |
| [DPA4 OMol25-100M](https://www.aissquare.com/models/detail?pageType=models&id=434&name=DPA4-OMol25_100M) | 20260820 | Nano, Mini, Neo, Air |
| [DPA4C OMol25-100M](https://www.aissquare.com/models/detail?pageType=models&id=435&name=DPA4C-OMol25_100M) | 20260820 | Nano, Mini, Neo, Air |

MatPES files in the inspected release use June dates, so they are outside
this August matrix. A model card's current installation instructions may
describe its September checkpoints; the checkpoint filename and hash
identify the actual release tested here.

## Method

`compatibility/validate_deepmd_release.py` exercises native fine-tuning,
freeze, ASE energy/force/stress inference, and deletion of the temporary
export. It preserves the matching JSON's model and optimizer settings,
including its full element order. It replaces the training data and sets
a one-epoch, batch-size-one budget and a small learning rate.
`--use-pretrain-script` loads the checkpoint's model description.

OMat24 tests use four slightly displaced Cu cells. OMol25 tests use
`--molecular`: four H2 geometries in a 20 angstrom periodic cell, with
explicit charge/spin values `[0, 1]` (neutral singlet) in both the training
arrays and ASE input. The cell separates periodic images and permits the
ASE stress interface to be exercised. Both datasets use EMT toy labels,
two training frames, one validation frame and one inference frame. These
tests establish execution compatibility, not scientific accuracy,
convergence, or suitability of EMT labels for training production models.

All computation runs through Slurm, with one GPU per task, four requested
CPUs, no explicit memory request, and at most two concurrent tasks. The
formal offline installation is
`/data/run01/scxm254/temper-review-20261027/deepmd-3.2.0`.
The installation audit reports DeepMD-kit 3.2.0, no modified or unverified
Python files in the audited backend/TEMPER distributions, and no broken
package requirements. The site CUDA driver stub is used only through
`LIBRARY_PATH` for linking exports, never through `LD_LIBRARY_PATH`.

Compiler caches use fresh job-local scratch unless the user supplied a
cache environment variable. Successful tests remove generated datasets,
trained checkpoints and exports after preserving concise results and the
effective configurations. Failure logs and work files remain in shared
storage for diagnosis; scratch cache paths recorded in a failure may cease
to exist when Slurm destroys the job-private temporary directory.

The new CPU tests mock DeepMD and GPU calls. They verify generated label
shapes, molecular conditioning, unchanged model/optimizer settings and
cleanup, and reject attempts to resume with different molecular conditioning.
Together with catalog and installation-checker tests, **63 targeted tests
passed**. Eight warnings came from ASE's deprecated NumPy shape assignment;
they did not affect the checks. These tests are separate from actual GPU results.

## Execution status

**All 16 checkpoint/JSON pairs passed** native fine-tuning, freeze, finite ASE
energy/force/stress inference, expected output shapes, and temporary export
cleanup. Every Slurm task completed with exit code `0:0` and exactly one GPU.
The first tasks began on 29 October at 15:13 China time; all results were
available by 16:44. The one-hour priority fallback was unnecessary.
`sinfo` listed `gpu_4090`, `gpu_5090`, `hp_4090` and `hp_5090`; no `ht_*`
partitions were present. No high-priority job was submitted.

The existing array was updated to permit `gpu_4090,gpu_5090`, without
submitting another copy. The official PyTorch/CUDA stack supports the
allocated GPUs. Actual GPU type is recorded per task below; this does not
claim every checkpoint was tested on both GPU models. Site-default memory
allocations were 40,000 MB on 4090 and 63,000 MB on 5090 nodes.

| Checkpoint | Array task | Job ID | GPU | Result |
| --- | --- | --- | --- | --- |
| DPA4-Air-OMat24-v20260805 | 1748943_0 | 1749020 | RTX 5090 | Pass |
| DPA4-Mini-OMat24-v20260805 | 1748943_1 | 1749021 | RTX 5090 | Pass |
| DPA4-Nano-OMat24-v20260805 | 1748943_2 | 1749066 | RTX 5090 | Pass |
| DPA4-Neo-OMat24-v20260805 | 1748943_3 | 1749071 | RTX 5090 | Pass |
| DPA4C-Air-OMat24-v20260819 | 1748943_4 | 1749092 | RTX 5090 | Pass |
| DPA4C-Mini-OMat24-v20260819 | 1748943_5 | 1749100 | RTX 5090 | Pass |
| DPA4C-Nano-OMat24-v20260819 | 1748943_6 | 1749119 | RTX 5090 | Pass |
| DPA4C-Neo-OMat24-v20260819 | 1748943_7 | 1749123 | RTX 5090 | Pass |
| DPA4-Air-OMol25-100M-v20260820 | 1748943_8 | 1749128 | RTX 5090 | Pass |
| DPA4-Mini-OMol25-100M-v20260820 | 1748943_9 | 1749144 | RTX 5090 | Pass |
| DPA4-Nano-OMol25-100M-v20260820 | 1748943_10 | 1749150 | RTX 5090 | Pass |
| DPA4-Neo-OMol25-100M-v20260820 | 1748943_11 | 1749166 | RTX 4090 | Pass |
| DPA4C-Air-OMol25-100M-v20260820 | 1748943_12 | 1749177 | RTX 4090 | Pass |
| DPA4C-Mini-OMol25-100M-v20260820 | 1748943_13 | 1749216 | RTX 5090 | Pass |
| DPA4C-Nano-OMol25-100M-v20260820 | 1748943_14 | 1749242 | RTX 4090 | Pass |
| DPA4C-Neo-OMol25-100M-v20260820 | 1748943_15 | 1748943 | RTX 4090 | Pass |

Exact task-to-checkpoint mappings, script contents and hashes are in
[`august-execution-20261029.json`](august-execution-20261029.json).
The fresh installation audit is
[`august-environment-20261029.json`](august-environment-20261029.json).
The [post-test audit](august-environment-after-20261029.json) has identical
package versions and no modified or unverified backend/TEMPER Python files.
The installed TEMPER wheel is unchanged from the previous formal-release
validation; its source manifest and wheel hash are linked in the execution
record. This matrix tests native DeepMD. The earlier TEMPER JobManager
round trip remains separate evidence, not a result of this array.

## Environment removal

The obsolete `/data/run01/scxm254/deepmd-kit` installation
(`3.2.1.dev41+g4e1479d8a`) was removed on 29 October at 07:00:11 UTC.
Its Conda history and package metadata were preserved before removal.
The retained formal prefix has no symbolic links into the removed prefix,
and its interpreter still runs after removal. Two stale Conda registrations
for already-absent temporary DeepMD installations were removed; the formal
prefix was registered explicitly.

MACE, MatterSim, SevenNet, NEP, the NEP overlay and the Miniconda base remain
installed and unchanged. The user explicitly confirmed that these working
environments must be preserved. The formal DeepMD prefix is now the
only retained DeepMD installation found in this account's deployment roots.

## Evidence and limits

The [per-checkpoint records](august-results-20261029.json) contain hashes,
commands, versions, energy values, shapes and cleanup results. The
[scheduler record](august-scheduler-20261029.txt) preserves actual allocations
and exit codes. The execution record preserves the exact GPU-tested validator
source: a later change only checks the molecular flag when resuming export
and is covered separately by CPU regression tests.

The backend emitted a warning about the absent optional
`nvalchemi-toolkit-ops` accelerator and used its native dense neighbor-graph
implementation. No backend source patch or scientific-setting workaround
was needed. TF32 performance suggestions were not applied. The tested
requirements include the official offline installer, PyTorch 2.12.0/CUDA
12.9, and vesin/vesin-torch 0.6.1; a generic version label alone is not the
reproduction recipe. Installer hashes remain in
[the original formal-release record](deepmd-release-validation.json).

These results cover the exact 16 downloaded pairs, not Plus/larger models,
other months, arbitrary replacement configurations, all molecular charge/spin
states, or production accuracy. OMol25 native tests explicitly supply
charge/spin data; they do not establish TEMPER's complete molecular-data
pipeline. The existing built-in OMat catalog entries now link their verified
AIS Square JSON/checkpoint pairs. The installation checker can recognize all
16 measured pairs by checkpoint hash, without treating installation checks
as GPU validation.

After all results and scheduler exit codes were verified and copied locally,
the task-owned remote directory was removed at 08:56:58 UTC on 29 October.
Successful validators had already removed generated datasets, exports and
owned scratch caches. The final cleanup removed shared model downloads,
raw training logs, scripts and archives. Local staging downloads and raw
logs were then removed; concise evidence and obsolete-environment provenance
remain in this documentation directory. See the
[cleanup record](august-cleanup-20261029.json) for exact paths and guards.
The formal DeepMD prefix and all other backend environments are retained.
