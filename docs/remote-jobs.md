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
| `input_utils.py` | Verification of packaged datasets and pretrained models. |
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
`create_bundles.py`, `job.py`, `run_pipeline.py`, and `split.py` each define
their own command arguments. `create_bundles.py` pairs the exported training
units with model specifications and writes bundles before submission.
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
runtime variables, scheduler flags and bundle settings free of credentials:
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
temper_bench job submit --resource-file cluster.yaml --keep-watching --interval 300
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
`uncertain`, because the scheduler may already have accepted the job. Run
`job submit` again to look for that existing job; TEMPER will not launch a new
attempt while the outcome remains unknown. Query or download errors are recorded
separately from calculation
failure; failed collection does not authorize remote cleanup.

## Local execution

Use the same submit, status, synchronize_remote, and cancel commands
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

## Input identity and refreshing bundles

Data splitting now records SHA-256 hashes of the actual exported training,
optional validation, and test files. These hashes participate in training-unit
identity and therefore bundle identity. Normal loading verifies those bytes;
an edit of the same file, even with unchanged size and timestamp, is detected.
The original record is not silently given a new identity.

Older units without hashes remain readable and usable for existing local
packaging. Remote submission requires their explicit conversion:

```console
temper_bench job refresh_bundle_identity bundle.json
```

This recomputes dataset hashes and the training-unit and bundle identities,
then replaces the bundle JSON. For `bundle.json` in the standard
`root/model/domain/bundle__...` layout, it also renames the containing bundle
directory to include the refreshed ID, preserving other files in that
directory. It holds the store lock and rejects an occupied destination rather
than overwriting it. Standalone JSON files keep their paths. The command prints
the final bundle path.

Directory renaming and JSON replacement are separate operations. If writing
fails before replacement, the command attempts to restore the original
directory name. Abrupt termination can interrupt this recovery; rerun refresh
using the file's current path to reconcile the directory name.
An existing attempt depends on its shared bundle. Refresh therefore rejects
bundles in bundle directories containing attempts, and bundles inside an
attempt directory. Create a separate bundle for a changed experiment.

There is no `--output` or
`--bundle-write-root` option for this command. Relative bundle paths and relative
training-unit `root_path` values use the current working directory.
Packaged file locations are cleared, so package the refreshed bundle again
before training or evaluation. Existing job history is not rewritten.
This also provides an explicit way to refresh a bundle after intentionally
changing its data files. For changed MLFF parameters
or pretrained weights, rebuild the MLFFSpec and bundle through their normal
builders. Pretrained weights are checked against their stored hash while copying.

In Python, `unit.copy_with_dataset_hashes()` returns a new hashed unit. It reads every
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

Create bundles from split results, then submit all bundle directories under
the same root:

```console
temper_bench create_bundles ./split_results --model mace mpa-0-medium --bundle-write-root ./bundles
temper_bench job submit --resource-file cluster.yaml --bundle-write-root ./bundles
```

Submission reads `*/*/bundle__*/bundle.json` beneath the root: one bundle per
model/domain/bundle directory. It excludes bundles inside `attempt_*` directories
and considers each bundle ID once. Individual bundle paths and JSON lists are
not command arguments. An empty root or a selection with no matches reports
that no bundles were found; it launches nothing.

**Run `job submit` again to continue an interrupted campaign.** Before launching
anything, TEMPER checks saved attempts and available remote evidence. It leaves
successful or running calculations alone, resumes interrupted preparation where
possible, and creates new attempts only when the previous outcome permits it.
An unreachable resource or an unknown submission outcome is reported and blocks
a new launch for that bundle; other bundles can still proceed. Exceptional
cases that cannot be recovered can be explicitly abandoned, as described below;
ordinary `submit` never makes that decision for you.

Supply `--resource-file` for bundles that have never been submitted. Existing
attempts are always checked and recovered using their saved resource settings.
New retries reuse those settings unless you supply a replacement resource file;
that replacement affects only new attempts. This lets you resume with:

```console
temper_bench job submit --bundle-write-root ./bundles
```

Use `--bundle-id`, `--domain`, or `--resource` to narrow the selection.
`--resource` is a resource name, not a profile filename. It matches the latest
saved attempt's resource, or the supplied profile for a bundle with no history.
Selection never allows an older failure to bypass a later active or successful
attempt.

To inspect the proposed actions without launching jobs:

```console
temper_bench job submit --bundle-write-root ./bundles --preview
# -p is the short form of --preview.
```

Preview contacts resources to check status and find existing jobs. It can save
those observations in local history, but does not upload, submit, cancel,
download results, or delete remote files. It reports the proposed action and
reason for each bundle, including missing resources and unresolved submissions.
The normal command checks again before acting, because jobs can change after a
preview. Preview cannot be combined with `--keep-watching`. `--json` provides
the same decisions in a machine-readable form.

`create_bundles` stores absolute dataset and pretrained-weight paths. Keep those
inputs available until submission packages them. Restarting from a different
working directory works when the same bundle root is supplied.

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
        bundle.json              # Bundle written by create_bundles
        attempt_001/
          bundle.json            # Temporary packaged bundle used for launch
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
files, bundle, scripts, small job records and eventual outputs remain. A retry
validates and copies the original inputs again into a new attempt.

## One-pass synchronization and watching

Submission returns after its checks and submission actions; it does not keep
watching by default. Each invocation creates at most one new attempt per bundle.
`submit --keep-watching` then monitors the confirmed attempts selected by that
invocation. It excludes blocked cases and does not add attempts launched later
by another command. It does not repeatedly restart failed calculations. Run
`submit` again when you want another eligible retry.
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
temper_bench job synchronize_remote --bundle-id FULL_UUID --keep-remote-files
```

`list` and `inspect` read local records. `status --refresh` queries and saves
status only: it does not upload, launch, cancel, download or delete anything.
This includes Bohrium: TEMPER bypasses dpdispatcher's status method that downloads
completed results. `synchronize_remote --keep-remote-files` collects and verifies
terminal results but retains remote working files. Run synchronization without
this option later to delete those files. Combining `--keep-remote-files` with
`--keep-watching` stops when every selected attempt has verified results,
without waiting for remote cleanup. The former `job download` command has been
removed. `logs` prints the paths of retained logs;
it does not implicitly download them. `--json` provides machine-readable records.

Each attempt separately records submission progress, computation outcome,
result collection, and remote cleanup. For example, a calculation can succeed
while its download is pending, or have verified results while deletion is pending.
Transient network errors do not turn a running calculation into a failed one.
The next synchronization retries the pending communication operation. Failed
calculations themselves are never relaunched by synchronization or watching.

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

## Safe retries, cancellation and interrupted submissions

There is one command for starting and continuing work: `job submit`. The former
`recover` and `resubmit` subcommands are removed. You do not need to decide which
kind of interruption occurred before running `submit` again.

| What TEMPER finds | What `submit` does |
| --- | --- |
| A successful attempt | Skips a new calculation; finishes outstanding collection or cleanup if needed. |
| A successful attempt alongside unfinished attempts | Reports the conflicting history for inspection; launches nothing. |
| A queued or running attempt | Keeps the existing attempt; does not submit another job. |
| Preparation or upload stopped before launch | Continues the same attempt where possible. |
| A launch began but its reply was lost | Looks for the existing job; resumes only when its identity is established. |
| The outcome remains unknown or the resource cannot be reached | Reports the problem and skips a new launch. |
| A confirmed failure or rejected submission | Completes required collection/cleanup, then creates one new attempt. |
| A cancelled attempt | Leaves it cancelled unless `--retry-cancelled` is supplied. |
| No attempt exists | Creates the first attempt using `--resource-file`. |

For example, inspect failures, preview, and then continue:

```console
temper_bench job list --failed
temper_bench job submit --domain my_domain --preview
temper_bench job submit --domain my_domain
# Use a different resource for new attempts, while checking old jobs where they ran:
temper_bench job submit --resource paratera --resource-file bohrium.yaml
```

Earlier attempts and their diagnostics remain in the local store. A new attempt
runs the whole bundle; it does not continue training from an intermediate
checkpoint. Source changes are rejected: create a new bundle identity when the
scientific inputs change. Collection or cleanup errors block a retry rather than
starting another calculation. Changing a queue, image, scratch directory, or
execution resource does not change the scientific identity.

These protections depend on retaining one shared local job store. Commands using
that store take its lock before making submission decisions. A copied or deleted
history cannot coordinate with the original history, and TEMPER cannot detect
unrelated jobs that you launch manually. Keep the store with the shared bundles
and their attempts; do not use independent copies to submit the same campaign.
Explicitly abandoning an unresolved attempt is an exception: it allows future
submission without proving that the old remote job has stopped.

To cancel a bundle's unfinished attempts:

```console
temper_bench job cancel --bundle-id FULL_UUID
```

Cancellation intent is saved before contacting the scheduler. Synchronization
waits for actual termination before collecting diagnostics and deleting remote
files. A later `submit` leaves the cancellation in effect unless you explicitly
request `--retry-cancelled`.

An unknown outcome is different from a confirmed failure. A scheduler can accept
a job just before the connection drops. TEMPER therefore checks saved job IDs
and submission receipts (files recording the scheduler's reply) before trying
another launch. Slurm can also be searched
by the attempt's unique working directory. An empty queue or missing accounting
record is inconclusive: the job may already have finished.

PBS recovery uses saved IDs or receipts. Bohrium uses its saved cloud identifiers.
Local execution uses the supervisor's receipt. If those records are insufficient,
TEMPER leaves the attempt unresolved rather than guessing that no job exists.
For a remote job you have independently identified, associate it with the exact
local attempt:

```console
temper_bench job attach --bundle-id FULL_UUID --attempt attempt_001 --job-id EXISTING_REMOTE_ID
temper_bench job submit --bundle-id FULL_UUID
```

`attach` records an existing job; it never launches one. It accepts unresolved
remote submissions and Bohrium upload drafts, not completed attempts or local
processes. Check the remote job's account, working directory, and other
identifying information before supplying its ID. Do not use the ID of an
unrelated calculation. The attempt name is required because one bundle may have
several saved attempts.

Supplied IDs must agree with saved IDs and any available remote receipts or
scheduler matches. For SSH, if no such evidence survives, attachment relies on
your independent verification. Bohrium additionally requires the job or draft
name returned by the service to equal `temper-<attempt token>`; its project ID,
when returned, must also match. If the service omits the identifying name,
TEMPER refuses the association rather than assuming ownership. An unresolved
attempt stays blocked until sufficient evidence is available or you explicitly
abandon it after investigating the remote job.

A Bohrium upload can expire before launch. When the draft ID was saved, continuing
the same attempt can recreate the unlaunched upload to obtain a fresh credential.
It does not rerun a calculation. Upload credentials and signed result URLs are
never written to the store. If the draft-creation response was lost, first find
the original draft in Bohrium and use `attach` with that draft ID and the exact
attempt. The next `submit` resumes its upload. Do not start a second blind
submission.

### Abandoning an unresolved attempt

Use `job abandon` only when TEMPER cannot establish an attempt's outcome and you
have decided to stop managing that attempt. It records that decision in local
history so later `submit` commands can create a replacement. **It does not stop
the remote calculation.** If the old job is still running, a later submission
can run the calculation twice. First inspect the scheduler or service and, when
possible, use `attach` to reconnect to the existing job instead.

A `blocked` submission decision does not by itself make an attempt eligible for
abandonment. TEMPER checks status and recovery evidence again, without changing
remote jobs or files. Only an unresolved submission, an unknown calculation
status, or an unresolved Bohrium upload draft can be selected. Attempts known
to be queued, running, successful, failed, or cancelled are excluded. Missing
resource profiles and pending result collection or cleanup are also not reasons
to abandon a job: fix the profile or retry synchronization for those cases.
A Bohrium upload with a saved draft ID is excluded when only the connection
fails; restore connectivity and continue the known draft instead.

Preview the eligible attempts first:

```console
temper_bench job abandon --preview
temper_bench job abandon --bundle-id FULL_UUID --preview
```

Without filters, `abandon` considers every eligible attempt in the selected
store. Optional `--bundle-id`, `--resource`, and `--domain` narrow the selection;
use the same `--bundle-write-root` as for submission. `--preview` (or `-p`)
reports the candidates without prompting or abandoning them. Status checks may
update the local observations used to explain eligibility.

To proceed interactively:

```console
temper_bench job abandon --bundle-id FULL_UUID
```

The command displays the selected attempts and the risk of a duplicate
calculation, then asks for confirmation with `[y/N]`. Pressing Enter declines.
For an unattended command, `--yes` is required:

```console
temper_bench job abandon --bundle-id FULL_UUID --yes
```

`--yes` skips only the prompt; it does not bypass eligibility checks. After
confirmation, TEMPER checks relevant local and remote state again while holding
the store lock. If the selection has changed, it stops without applying the
abandonment; run a fresh command to inspect and confirm the current selection.
Changes to statuses, execution settings, or saved job identifiers require a new
review. A new check time or an updated error message alone does not invalidate
confirmation, but the attempt must still be eligible. Records are written one
at a time: a local write failure can leave some attempts abandoned and others
unchanged. Inspect the history before retrying after such an error.

Abandonment never submits, cancels, or deletes anything. It retains the existing
job record, statuses, identifiers, diagnostics, and files, and adds the
`abandoned_at` timestamp and `abandoned_reason`. `job list` and `job inspect`
continue to show the attempt, marked as abandoned. Routine submission,
monitoring, cancellation, and cleanup stop managing it. Investigating or removing
any remaining remote work becomes your responsibility.

Run `job submit` separately when ready to start again. It can reuse the resource
settings from the last abandoned attempt unless you supply another profile.
Other active or successful attempts for the same bundle still prevent an
unnecessary new calculation. Do not abandon a known job to work around a
configuration error or a failed download.

## Validation boundary

For native Windows transfer, explicit Slurm task/CPU fields, confirmed rejection
recovery through `job submit`, optional `sbatch --test-only`, and job-local
cache configuration, see [deployment compatibility](deployment-compatibility.md).
That guide distinguishes unit coverage from recorded GPU and JobManager tests.

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
`JobManager.submit_bundles()` implements the CLI's check-and-continue behavior
and returns a `SubmissionDecision` for each selected bundle. Each decision
explains the action, reason, chosen resource, and attempt path when available.
`JobManager.watch_submissions()` monitors the confirmed attempts from those
decisions without starting retries. The older Python methods `submit`,
`recover`, and `resubmit` remain available for callers that explicitly need
their separate operations; the CLI uses the unified workflow.
`get_dataset_and_model_hashes(bundle)` excludes configuration files and scripts;
`verify_dataset_and_model_hashes(bundle, root)` checks only those copied files.
`get_download_files(bundle)` returns exact download names and required-success
names. `JobRecord.relative_source_bundle_path` references the shared bundle, normally
`../bundle.json` relative to the attempt directory. The full bundle is stored
once in the parent bundle directory, not repeated in each job record.
The first submission fills the shared bundle's `files` field with the packaged
input/output locations. Every attempt uses that same layout; changing packaging
settings so they produce different locations is rejected before copying inputs.
Loading checks that the shared bundle has the attempt's bundle ID. A missing
bundle or a different identity is an error. Keep the shared bundle with its
attempts; moving their containing directory together preserves the reference.

After a confirmed remote submission, TEMPER removes the local attempt's
`bundle.json`. The uploaded copy remains available to the remote runner until
remote cleanup. Interrupted or uncertain submissions retain the local copy
until recovery confirms launch. Local execution retains it until the calculation
has finished and completion cleanup runs, because the local runner reads this
file after submission returns. Result collection, cancellation, and status
queries read the shared bundle, including its file locations. Neither the full
bundle nor its file-location mapping is stored in `job.json`.

Schema classes now live in `job_management.schemas`; the old `records` and
`resources` modules retain compatibility imports. Version-3 job records use
`relative_source_bundle_path` and `downloaded_file_hashes`. The `manifest` spelling remains
accepted for download hashes. Older version-1/2 records containing embedded
bundles or file-location mappings require conversion before use; they are
rejected rather than silently losing the information they contain. This command
does not automatically rewrite existing history.
Old saved Bohrium field names and frozen `environment`
remain readable. New profile files should use the names documented above.
Per-attempt `job.json` remains authoritative. Storage indexing is deferred pending
benchmarks of large histories; no unified JSONL database has been introduced.
The optional `abandoned_at` and `abandoned_reason` fields retain an explicit
decision to stop managing an unresolved attempt. Existing records without these
fields are treated as not abandoned. Abandonment does not rewrite the recorded calculation
outcome or remove its history.


### Retrying a scheduler rejection

`job submit` handles both failed calculations and confirmed rejected
submissions. No bundle ID is required: use `--preview` to see all eligible
bundles and skip reasons, and `--resource-file corrected.yaml` to apply corrected
resources. Optional `--bundle-id`, `--resource`, and `--domain` filters narrow
the selection. A newer attempt outside the filters still prevents replaying
an older failure. Rejected attempts retain their submission diagnostics and
are cleaned before retrying; they are not recorded as failed calculations.
Unknown outcomes are checked for an existing remote job before any new launch.
An unresolved outcome or cleanup failure prevents launch and is reported without
stopping other selected bundles.

`slurm_preflight: true` optionally checks the scheduler request with
`sbatch --test-only`, then submits the real job if the check passes. It does not
reserve a GPU or test the calculation. The former profile key
`slurm_test_only` remains readable; newly saved profiles use `slurm_preflight`.
