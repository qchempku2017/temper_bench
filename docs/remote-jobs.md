# Remote submissions and persistent job history

TEMPER can submit one MLFF bundle per job to an SSH-accessible Slurm or PBS
cluster, or to Bohrium. Paratera uses an ordinary Slurm resource profile with
SSH key-file authentication. No Paratera-specific Python API is used.

Install the local submission dependencies:

```console
pip install 'temper-bench[remote]'
# Include the Bohrium SDK when using Bohrium:
pip install 'temper-bench[bohrium]'
```

The integration targets **dpdispatcher 1.0.3** and **bohrium-sdk 0.15.0**.
Remote compute environments need TEMPER, Bash, and the selected MLFF software.
They do not need dpdispatcher, MongoDB, or a permanently running TEMPER service.
The local process uploads inputs and asks the remote scheduler to run TEMPER.

## Python package organization

Remote job functionality lives in `src/temper/job_management/`, with its CLI
in `src/temper/entrypoints/job.py` alongside `run_pipeline.py` and `split.py`:

| Module | Purpose |
| --- | --- |
| `manager.py` | Submission, periodic monitoring, restart recovery, cancellation, and explicit retries. |
| `records.py` | The `JobRecord` schema for one submission attempt. |
| `resources.py` | Remote machine settings and MLFF-specific software environments. |
| `store.py`, `storage_utils.py` | Local JSON history, attempt markers, atomic writes, and process locking. |
| `paths.py` | Bundle roots and readable bundle/attempt directory names. |
| `inputs.py` | Verification of packaged datasets and pretrained models. |
| `collection.py` | Selection and verification of downloaded outputs and diagnostics. |
| `dispatcher.py` | SSH/Slurm/PBS and Bohrium operations through dpdispatcher. |

For Python callers, import the objects from their defining modules:

```python
from temper.job_management.manager import JobManager
from temper.job_management.resources import RemoteResource
from temper.job_management.store import JobStore
```

The main CLI registers each command using its entrypoint module:
`job.py`, `run_pipeline.py`, and `split.py` each define their own subparser.
Bundle writers use the job-management package's path
and storage helpers, and the execution runtime uses its packaged-input check.
MLFF adapters declare which model-specific outputs to retain. Data-file hashing
remains in `temper.utils.file_hashing` because data splitting also uses it.

## Resource profiles and software environments

Save this as `cluster.yaml`, replacing the example values:

```yaml
name: paratera
backend: Slurm                 # PBS is also supported
hostname: login.example.org
username: your-account
port: 22
key_filename: keys/id_ed25519   # Relative to this YAML file
known_hosts: keys/known_hosts  # Previously verified host keys
passphrase_env: TEMPER_SSH_PASSPHRASE  # Omit for an unencrypted key
remote_root: /scratch/your-account/temper
timeout: 30                    # Connection/command timeout in seconds
resources:
  number_node: 1
  cpu_per_node: 8
  gpu_per_node: 1
  queue_name: gpu
  custom_flags:
    - '#SBATCH --time=12:00:00'
    - '#SBATCH --account=your-allocation'
environments:
  mace:
    conda_init: /opt/miniconda3/etc/profile.d/conda.sh
    conda_environment: temper-mace
  dpa4:
    modules: [cuda/12.4]
    conda_init: /opt/miniconda3/etc/profile.d/conda.sh
    conda_environment: temper-deepmd
```

`environment` provides the default setup. An entry under `environments` replaces
that setup completely for its MLFF family; it does not merge individual fields.
Optional `variables` supplies non-secret environment values. Setup runs inside
the task with failure checking: a failed module load or conda activation stops
the task and is recorded in its diagnostics.

For PBS, use `backend: PBS`, the appropriate queue, and `#PBS` custom flags.
Site-specific resource syntax still needs checking on the cluster. TEMPER uses
`squeue` and `sacct` for Slurm observation, and full `qstat` records for PBS.
An exiting PBS job is still active. If a scheduler purges a job before returning
terminal evidence and no completion tag exists, TEMPER reports an inconclusive
state and preserves remote data. It does not interpret a broken query as failure.

SSH supports RSA/ECDSA/Ed25519 keys through Paramiko. Unknown or changed host
keys are rejected. If `known_hosts` is omitted, Paramiko reads the user's normal
known-hosts file. Have the cluster administrator verify new host fingerprints
before adding them. `proxy_command` may specify an OpenSSH ProxyCommand for a
jump host. Set passphrases in the named local environment variable; do not put
them in YAML. On POSIX, for example, `export TEMPER_SSH_PASSPHRASE=...` makes the
value available to subsequent commands. A background watcher needs the same
credential environment when restarted.

A Bohrium profile looks like this:

```yaml
name: bohrium-backup
backend: Bohrium
project_id: 12345
machine_type: YOUR_BOHRIUM_MACHINE_TYPE
platform: ali
access_key_env: BOHRIUM_ACCESS_KEY
resources:
  number_node: 1
  cpu_per_node: 8
  gpu_per_node: 1
environments:
  mace:
    image: YOUR_REGISTRY/temper-mace:YOUR_FIXED_TAG
  dpa4:
    image: YOUR_REGISTRY/temper-deepmd:YOUR_FIXED_TAG
```

Each image must already contain the corresponding TEMPER runtime and MLFF
dependencies. Credentials are resolved locally at connection time. The saved
profile contains the environment variable's **name**, never its value. Keep
runtime variables, scheduler flags and recipe settings free of credentials:
those settings are intentionally retained for reproducibility.

## Input identity and upgrading old recipes

Data splitting now records SHA-256 hashes of the actual exported training,
optional validation, and test files. These hashes participate in training-unit
identity and therefore bundle identity. Normal loading verifies those bytes;
an edit of the same file, even with unchanged size and timestamp, is detected.
The original record is not silently given a new identity.

Older units without hashes remain readable and usable for existing local
packaging. Remote submission requires their explicit conversion:

```console
temper_bench job upgrade_bundle old_bundle.json --output hashed_bundle.json
```

The destination must be new. This also provides an explicit way to rebuild a
recipe after intentionally changing its data files. For changed MLFF parameters
or pretrained weights, rebuild the MLFFSpec and bundle through their normal
builders. Pretrained weights are checked against their stored hash while copying.

In Python, `unit.with_dataset_hashes()` returns a new hashed unit. It reads every
distinct file once. Export, packaging and submission share short-lived hash
caches within the operation; later independent loads read the files again.
For a JSON list of units, `temper.schemas.load_training_units(path)` also shares
hashes within that one load, so a test dataset used by many units is read once.
Copying independently hashes the bytes written so a source edit during copying
cannot silently reuse the old identity. SHA-256 reads in bounded chunks, rather
than loading an entire dataset into memory.

Historical inspection without the original datasets is explicit:

```python
bundle = MLFFTrainBundle.model_validate_json(
    saved_text, context={"verify_files": False}
)
```

This checks the metadata identities but does not establish that input files are
available or unchanged. Submission verifies the sources again. A remote runtime
uses this metadata mode and then verifies the **copied** inputs relative to
`bundle.json`; it never tries to open the local submission machine's paths.

## Submit and locate attempts

Save a local recipe as JSON, then submit it:

```python
from pathlib import Path
Path("recipe.json").write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
```

```console
temper_bench job submit recipe.json --resource-file cluster.yaml --bundle-write-root ./bundles
```

Pass several recipe files, or a JSON list such as `["recipes/a.json",
"recipes/b.json"]`; list entries resolve relative to the list file. Dataset roots
inside recipes retain their documented current-working-directory interpretation.
TEMPER freezes absolute source paths when saving an attempt, so restarting from
a different working directory works when the same store root is supplied.

`--bundle-write-root` overrides `DEFAULT_BUNDLE_WRITE_ROOT`, whose fallback is
`./bundles`. The root environment variable is read when the operation starts.
Python packaging accepts the matching keyword `bundle_write_root=...`.
An explicit `target_dir` still means exactly that new destination and cannot be
combined with the root keyword.

The automatic layout is:

```text
bundles/
  <pretrained_model_name>__<model_version>/
    <domain>/
      bundle__<grouping>__<group>__<split>__<mode>__n<N>__repeat<R>__<full_bundle_id>/
        attempt_001/
          bundle.json
          job.json
          events.jsonl
          run.sh
          temper_job.sh
          training_outputs/
          testing_outputs/
          stdout.log
          stderr.log
          success                 # Or failed / cancelled
        attempt_002/
```

Names are sanitized for Windows and POSIX. Long labels are shortened with a
digest; the full bundle ID remains in the directory name. On Windows, Python
uses the `\\?\` long-path prefix internally so nested model outputs do not hit
the usual 260-character limit. Use a short root if another program cannot open
long paths. An operating-system lock protects attempt allocation and store
mutations; a second writer reports that the store is busy instead of racing.
Use a local filesystem for the store, since network filesystem lock and durability
semantics differ.

Inputs are **ordinary copies** in `pretrained_models/`, `training_inputs/` and
`testing_inputs/`. After confirmed upload and a durable recovery record, TEMPER
removes only these copied datasets/weights and its upload archives. Your source
files, recipe, scripts, small job records and eventual outputs remain. Explicit
resubmission validates and copies the original inputs again into a new attempt.

## One-pass synchronization and watching

Submission returns after launching; it does not keep watching by default.
To check jobs once, download terminal results, verify them, and clean up:

```console
temper_bench job synchronize_remote --bundle-write-root ./bundles
```

To keep doing this every minute:

```console
temper_bench job synchronize_remote --bundle-write-root ./bundles --keep-watching --interval 60
# On a POSIX submission host:
nohup temper_bench job synchronize_remote --bundle-write-root /absolute/bundles --keep-watching > temper-monitor.log 2>&1 &
```

`--keep_watching` is an alias. It also works on `job submit`. Watching remains a
foreground loop; `nohup`, a terminal multiplexer or your service manager decides
how to keep that process alive. Ctrl+C stops monitoring without cancelling jobs.
After power loss, repeat the synchronization/watch command with the same root.
No database recovery or calculation restart is needed for already submitted jobs.

Useful inspection commands:

```console
temper_bench job list
temper_bench job list --failed --domain my_domain
temper_bench job inspect --bundle-id FULL_UUID
temper_bench job logs --bundle-id FULL_UUID
temper_bench job status --refresh --resource paratera
temper_bench job download --bundle-id FULL_UUID
```

`list` and `inspect` read local records. `status --refresh` queries and saves
status only: it does not upload, launch, cancel, download or delete anything.
This includes Bohrium: TEMPER bypasses dpdispatcher's status method that downloads
completed results. `download` collects and verifies terminal results but leaves
remote deletion for synchronization. `logs` prints the paths of retained logs;
it does not implicitly download them. `--json` provides machine-readable records.

Each attempt separately records submission progress, computation outcome,
result collection, and remote cleanup. For example, a calculation can succeed
while its download is pending, or have verified results while deletion is pending.
Transient network errors do not turn a running calculation into a failed one.
The next synchronization retries the pending communication operation. Failed
calculations themselves are never automatically relaunched.

## Retained results and remote deletion

TEMPER retains the final trained model, test predictions, test metadata/summary,
training logs, available scheduler/task diagnostics, and selected generated
configs, scripts and metrics. Adapter output declarations control this selection.
Converted training datasets, pretrained copies and intermediate checkpoint
directories are excluded. A successful job must have every expected prediction,
metadata/summary and final-model file, and those files must be nonempty.

SSH downloads are compared to remote SHA-256 digests. Bohrium result archives are
read through ZIP CRC validation; only selected safe relative filenames are
extracted. Downloads use partial files and then replace their final destinations.
The collected manifest records SHA-256 values and is checked again before deletion.
This verifies transfer completeness, not the scientific accuracy of predictions.

A stopped failed job may legitimately have no final model or predictions.
Available diagnostics and selected outputs are still collected and verified,
then the remote attempt is deleted. A transport failure or an unavailable Bohrium
result archive is not evidence that files are absent: cleanup stays pending.
TEMPER conservatively retains the remote job if no diagnostics/results can be
collected. Resolve provider archive availability before deleting those jobs.

For SSH, deletion is limited to the unique attempt directory under `remote_root`,
checked against its ownership token. For Bohrium, TEMPER calls the actual job
deletion endpoint and requires a successful response. dpdispatcher's context
`clean()` alone does **not** delete a cloud job. A failed deletion remains pending
and does not invalidate verified successful results.

`job.json` is authoritative. `failed` is created on confirmed calculation failure;
`success` is created only after successful calculation **and verified results**.
Synchronization repairs missing or incorrect marker files from the record.

## Explicit retries, cancellation and emergency recovery

Inspect failures first, then retry eligible bundles together:

```console
temper_bench job list --failed
temper_bench job resubmit --failed --domain my_domain --dry-run
temper_bench job resubmit --failed --domain my_domain
# Retry eligible failures using the backup resource:
temper_bench job resubmit --failed --resource paratera --resource-file bohrium.yaml
```

Each bundle is retried at most once per command. Earlier attempts remain intact.
There is no training/evaluation-stage resume: a new attempt runs the whole recipe.
Source changes are rejected; explicitly rebuild a new identity for changed science.
An active, uncertain or incompletely collected attempt blocks another attempt.
Any successful attempt blocks that bundle permanently within the retained store,
including when the command selected an older failure. Changing a queue, image,
scratch directory or execution resource does not create a new scientific identity.
Keep or back up the store: a separate/deleted history cannot know about earlier
successes.

```console
temper_bench job cancel --bundle-id FULL_UUID
temper_bench job recover --bundle-id FULL_UUID
```

Cancellation intent is saved before contacting the scheduler. Synchronization
waits for actual termination before downloading diagnostics and deleting data.

Recovery is distinct from resubmission. For a crash before a launch was attempted,
it can finish the original upload/submission. If a launch response was lost,
TEMPER reads its remote SSH receipt or saved cloud ID and attaches to that job;
it never assumes the launch failed merely because the local ID is missing.
If no receipt survives, find the existing job on the remote resource first:

```console
temper_bench job recover --bundle-id FULL_UUID --job-id EXISTING_REMOTE_ID
```

The supplied ID is your assertion that this is the same attempt. Do not provide
an unrelated job ID. An unresolved attempt stays blocked instead of risking a
duplicate calculation. This is especially important after a lost cloud creation
response; the cloud job name contains the saved attempt token for lookup.

A Bohrium upload can expire before launch. When its draft ID was saved, recovery
deletes the unlaunched draft and recreates the upload to obtain a fresh credential.
It does not rerun a calculation. Upload credentials and signed result URLs are
never written to the store. If even the draft-creation response was lost, provide
that draft ID explicitly before recovery; do not start a second blind submission.

## Validation boundary

Offline tests exercise real dpdispatcher serialization/script interfaces with
mocked remote connections and simulated interrupted operations. They do not
consume cluster allocations or launch paid Bohrium jobs. Before a production
campaign, run one small bundle on each actual resource to check site queue
syntax, installed runtime versions, image contents, accounting visibility,
provider result availability and deletion permissions.
