# Local and remote submissions with persistent job history

TEMPER can submit one MLFF bundle per job to an SSH-accessible Slurm or PBS
cluster, to Bohrium, or to a local POSIX machine. Paratera uses an ordinary Slurm resource profile with
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
| `schemas/records.py` | The `JobRecord` schema for one submission attempt. |
| `schemas/resources.py` | Remote machine settings and MLFF-specific software environments. |
| `store.py`, `storage_utils.py` | Local JSON history, attempt markers, atomic writes, and process locking. |
| `paths.py` | Bundle roots and readable bundle/attempt directory names. |
| `inputs.py` | Verification of packaged datasets and pretrained models. |
| `collection.py` | Selection and verification of downloaded outputs and diagnostics. |
| `local.py` | Detached local execution, durable receipts, and cancellation. |
| `dispatcher.py` | SSH/Slurm/PBS and Bohrium operations through dpdispatcher. |

For Python callers, import the objects from their defining modules:

```python
from temper.job_management.manager import JobManager
from temper.job_management.schemas.resources import RemoteResource
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
key_filename: ~/.ssh/id_ed25519 # Original key on the submission machine
known_hosts: ~/.ssh/known_hosts  # Previously verified host keys
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
    conda_environment: temper-mace
  dpa4:
    modules: [cuda/12.4]
    conda_environment: temper-deepmd
```

Specify an entry under `environments` for every MLFF family you submit. An empty
entry (`mace: {}`) explicitly uses the job's existing environment. Missing family
entries are errors. The selected environment is copied into the attempt's internal
`resolved_environment` field; changing the original profile cannot change recovery.
User profile files do not accept `environment` or `resolved_environment`.

Conda activation initializes Bash with `conda shell.bash hook`; the executable
must be available in the job's PATH. Optionally set `conda_init` to an absolute
runner path to `conda.sh`. TEMPER does not edit shell startup files. Alternatively,
use `venv_path: /home/your-account/venvs/temper-mace`; TEMPER sources its
`bin/activate`. Conda and venv settings are mutually exclusive.
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
before adding them.

**Before the first TEMPER connection, manually log in once with OpenSSH from
the machine and user account that will run TEMPER to populate `known_hosts`.**
This step can be skipped only if a verified entry for the target SSH server is
already available in a host-key file that TEMPER loads. TEMPER does not prompt
to trust an unknown host or automatically add its key, even when your user public
key has already been registered in the cluster's web console.

For example, on Linux or WSL:

```console
ssh -i ~/.ssh/id_ed25519 -p 22 your-account@login.example.org
```

On Windows CMD, an equivalent example is:

```console
ssh -i "%USERPROFILE%\.ssh\id_ed25519" -p 22 your-account@login.example.org
```

Replace the private-key path, port, remote account, and hostname with your actual
settings. Use the same hostname and port as the TEMPER resource profile; an entry
saved for an IP address may not match a connection made through a DNS name.
Non-default ports are recorded as `[hostname]:port`. For password authentication,
omit `-i`. If a jump host is required, use the corresponding connection settings
for this manual login as well.

When OpenSSH asks whether to trust the server, compare the displayed fingerprint
with the one supplied by the cluster administrator or platform, then enter `yes`
only if they match. With the default OpenSSH configuration, this saves the server
key in `~/.ssh/known_hosts` on Linux/WSL or `%USERPROFILE%\.ssh\known_hosts` on
Windows. Complete the login and then run `exit` to return to your local shell.
Point the profile's `known_hosts` at that file. If you use a custom file, pass
`-o UserKnownHostsFile="/path/to/known_hosts"` to OpenSSH and configure TEMPER to
read the same file. An explicitly configured file must exist and be readable.

Windows and WSL normally have separate user home directories and `known_hosts`
files. Repeat this preparation in the environment that will run TEMPER, or
explicitly configure an accessible file containing the verified entry. The
server key in `known_hosts` verifies the server's identity; your public key in
the cluster console authorizes your account. They serve different purposes.
If a previously recorded server key changes, confirm the replacement with the
administrator before updating the entry.

`proxy_command` may specify an OpenSSH ProxyCommand for a
jump host. Set passphrases in the named local environment variable; do not put
them in YAML. On POSIX, for example, `export TEMPER_SSH_PASSPHRASE=...` makes the
value available to subsequent commands. A background watcher needs the same
credential environment when restarted.

Private keys are never copied into submission attempts or uploaded. Absolute
paths and `~` are recommended. Relative `key_filename` and `known_hosts` paths
resolve against the profile file's directory on the submission machine, so every
attempt references the same original file. If no key is configured, an explicit
`password_env: TEMPER_SSH_PASSWORD` enables password authentication. A configured
key that fails does not silently fall back to a password. Credentials are read
from the submitting process's environment only when connecting.

Configure `remote_root` explicitly, for example `/home/your-account/temper_run`
or a shared scratch directory visible to compute nodes. An omitted value falls
back to `/tmp/temper-<user-digest>` with a warning at submission: `/tmp` may be
cleared or may not be shared with compute nodes. Neither `$HOME` nor `~` is
expanded in remote_root; provide an absolute remote path. Connection attempts
are bounded to five total attempts, ten seconds apart for transient failures.
Authentication and host-key failures are not retried. Remote submission commands
are never automatically replayed after a lost response.

A Bohrium profile looks like this:

```yaml
name: bohrium-backup
backend: Bohrium
bohrium_project_id: 12345
bohrium_machine_type: YOUR_BOHRIUM_MACHINE_TYPE
bohrium_platform: ali
bohrium_access_key_env: BOHRIUM_ACCESS_KEY
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

## Slow or intermittently responsive SSH clusters

For Slurm/PBS clusters with slow command responses or intermittent transfer
stalls, configure both the operation timeout and the polling interval. A quick
SSH login does not imply that scheduler queries will return quickly: delays can
also come from the cluster's gateway, scheduler, or shared filesystem.

Add the following to your resource profile as an initial setting, then adjust
it to the observed response times:

```yaml
timeout: 120
```

`timeout` defaults to 30 seconds and accepts an integer of at least 1, with no
configured upper limit. For SSH resources, it controls connection, banner and
authentication waits, command-channel opening and blocking reads/writes, and
SFTP channel reads/writes. It is not a total deadline for a calculation, command,
or file transfer. A download that continues making progress can take much longer
than this value; a stalled read or a remote checksum command that produces no
output for too long can still time out. Increasing this value also increases
how long a failed operation may take to report an error.

**Resource settings are saved with each attempt.** Editing `timeout` in the
original YAML affects new attempts, but does not automatically change the value
used to query or download an existing attempt. Those operations use the saved
resource settings.

To reduce repeated queries, increase the watcher's polling interval, for example
to five minutes:

```console
temper_bench job synchronize_remote --resource paratera --keep-watching --interval 300
```

Replace `paratera` with the saved resource name. To apply the same interval when
starting a watcher immediately after submission:

```console
temper_bench job submit bundle.json --resource-file cluster.yaml --keep-watching --interval 300
```

The Python equivalent is `manager.watch(interval=300, resource="paratera")`.
The interval defaults to 60 seconds and must be positive. It is a watcher
argument, not a resource YAML field, and can be changed when restarting a watcher
for existing attempts. Without `--keep-watching`, synchronization runs once.

The watcher finishes a synchronization pass before sleeping for the requested
interval. A 45-second pass followed by `--interval 300` starts the next pass
about 345 seconds after the previous one began; one watcher does not overlap
passes when a query is slow. Synchronization can include result downloads and
cleanup as jobs finish, so a pass can take longer than a status query alone.
Increasing the interval reduces query frequency but does not extend any
individual request's timeout, and it delays detection of completed jobs.

SSH keepalive is separate from task polling. TEMPER currently configures a
30-second keepalive interval to help preserve idle SSH connections; it does not
run scheduler queries or impose a 30-second job deadline. This interval and the
connection retry policy (up to five attempts, ten seconds apart for eligible
transient errors) are not exposed as resource settings.

The current timeout handling has limits. TEMPER does not override every
Paramiko timeout, including its internal key-exchange waits. Some DPDispatcher
status-query paths wait for a command's exit status without a total deadline;
raising `timeout` therefore does not guarantee that every operation will finish
or fail within that many seconds. A lost submission response leaves the attempt
`uncertain`, because the scheduler may already have accepted the job. Use
recovery to establish the existing job's identity rather than blindly submitting
it again. Query or download errors are recorded separately from calculation
failure; failed collection does not authorize remote cleanup.

## Local execution

Use the same submit, status, synchronize, cancel, recover, and resubmit commands
with this profile on a POSIX host (for example Linux or a configured WSL project):

```yaml
name: workstation
backend: Local
environments:
  mace:
    conda_environment: temper-mace
```

Native Windows local execution is not supported: the packaged MLFF commands
require POSIX Bash. Windows can still submit to SSH and Bohrium resources.
Local jobs use a detached supervisor and run inside their attempt directory.
Stopping the submitting CLI or watcher leaves the job running. Dataset/model
copies remain until calculation termination and verified collection. Collection
hashes retained files in place; cleanup then removes copied inputs and unretained
training-output directories. Configurations, final weights, and results remain.
Cancellation requests are handled by the live supervisor, which terminates its
calculation process group. Recovery reads a durable local receipt; it never
blindly restarts a process. A lost supervisor without an exit receipt remains
unknown after its heartbeat expires and requires investigation.

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
`./bundles`. Like other values in `utils.defaults`, this environment variable
is read once at import time; set it before starting Python.
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
configs, scripts and metrics. Adapter output declarations list exact relative
filenames; wildcards and recursive native-output downloads are not supported.
Scheduler stdout/stderr use `scheduler.stdout.log` and `scheduler.stderr.log`
(PBS may combine them). Optional diagnostics may be absent or empty; successful
scientific outputs are checked separately. Native checkpoint directories and
unlisted model variants are never selected.
Converted training datasets, pretrained copies and intermediate checkpoint
directories are excluded. A successful job must have every expected prediction,
metadata/summary and final-model file, and those files must be nonempty.

SSH downloads are compared to remote SHA-256 digests. Bohrium result archives are
read through ZIP CRC validation; only selected safe relative filenames are
extracted. Downloads use partial files and then replace their final destinations.
The record field `downloaded_file_hashes` stores SHA-256 values and is checked
again before deletion.
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

## Final-epoch model policy

Evaluation and collection use the same final-epoch weights, never the validation
best checkpoint. MACE saves every epoch, disables EMA/SWA and early stopping,
and verifies the final epoch checkpoint exists before evaluation. MatterSim uses
its last checkpoint with early stopping disabled; TEMPER's process-local launcher
sets EMA decay to zero and verifies its recorded last epoch. SevenNet copies the
checkpoint named for the configured final epoch. NEP uses `nep_final.txt` and
checks the final epoch in `loss.out`; `nep_best.txt` is not retained. DeepMD retains
the configured final training checkpoint after its `numb_epoch` budget.
Intermediate checkpoints are not downloaded. Missing final outputs fail the job
or result verification; an early checkpoint is never substituted.

The native contracts were checked against MACE 0.3.16 (`tools/train.py` and
`cli/run_train.py`), MatterSim 1.2.5 (`forcefield/potential.py`), and TorchNEP 1.0.2
(`train.py`). Full GPU training and site-specific scheduler behavior must still
be exercised on the intended compute resources.

## Python names and saved-history compatibility

`JobManager.list_attempts()` reads filtered local history without remote calls.
`get_dataset_and_model_hashes(bundle)` excludes configuration files and scripts;
`verify_dataset_and_model_hashes(bundle, root)` checks only those copied files.
`get_download_files(bundle)` returns exact download names and required-success
names. `JobRecord.source_bundle` contains the original serialized bundle with
source-machine paths and hashes, used to reconstruct retries. It differs from
the packaged `bundle.json`, which contains attempt-relative paths.

Schema classes now live in `job_management.schemas`; the old `records` and
`resources` modules retain compatibility imports. Saved `recipe` and `manifest`
keys are accepted when reading old records; new writes use `source_bundle` and
`downloaded_file_hashes`. Old saved Bohrium field names and frozen `environment`
remain readable. New profile files should use the names documented above.
Per-attempt `job.json` remains authoritative. Storage indexing is deferred pending
benchmarks of large histories; no unified JSONL database has been introduced.
