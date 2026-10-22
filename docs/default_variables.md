# Default variables

[Back to the project overview](../README.md) · [MLFF bundles](mlff-bundles.md) · [Data splitting](data-splitting.md)

`temper.utils.defaults` exports the constants below. Each value is read once
when the module is imported. Set environment variables before importing TEMPER
(or before starting the CLI process); changing the environment afterward does
not update an already imported module.

## Data and output paths

These string constants use the environment variable with the same name. For
the older general-purpose paths, an unset variable uses the listed default and
an explicitly empty value remains an empty string.

| Constant | Type | Default | Purpose |
| --- | --- | --- | --- |
| `DEFAULT_DATA_DIR` | `str` | `./data` | Root containing source data domains. |
| `DEFAULT_SPLIT_RESULTS_DIR` | `str` | `./split_results` | Root for exported split data and TrainingUnits. |
| `DEFAULT_SPLIT_CONFIG_FILE` | `str` | `split_config.json` | Configuration read by the split CLI when no path is supplied. |
| `DEFAULT_METADATA_FILE` | `str` | `metadata.json` | Metadata filename inside each source domain. |
| `DEFAULT_GROUPED_DOMAIN_FILE` | `str` | `grouped_domains.json` | Persisted grouping output filename. |
| `DEFAULT_SPLIT_GROUPS_FILE` | `str` | `split_groups.json` | Persisted split output filename. |
| `DEFAULT_TRAINING_UNITS_FILE` | `str` | `training_units.json` | Persisted TrainingUnit output filename. |

## MLFF paths

The job store and automatic bundle layout use `DEFAULT_BUNDLE_WRITE_ROOT`
(default `./bundles`). This string constant is read once when `utils.defaults`
is imported. `--bundle-write-root` or the Python `bundle_write_root` keyword takes
precedence. Relative values resolve against the current working directory.
See [Remote jobs](remote-jobs.md) for the model/domain/bundle/attempt layout.

`DEFAULT_MLFF_PRETRAINED_MODELS_DIR` is a string source root. Its default is
`./pretrained_models`, and it may be replaced by any relative or absolute path
through the environment variable of the same name.

The following strings name directories inside every written submit folder.
Their same-named environment variables may replace the defaults:

| Constant | Type | Default |
| --- | --- | --- |
| `DEFAULT_MLFF_MODELS_DIR` | `str` | `pretrained_models` |
| `DEFAULT_MLFF_TRAINING_INPUTS_DIR` | `str` | `training_inputs` |
| `DEFAULT_MLFF_TRAINING_OUTPUTS_DIR` | `str` | `training_outputs` |
| `DEFAULT_MLFF_TESTING_INPUTS_DIR` | `str` | `testing_inputs` |
| `DEFAULT_MLFF_TESTING_OUTPUTS_DIR` | `str` | `testing_outputs` |

Submit-directory values must be non-empty, normalized relative POSIX paths.
They may contain subdirectories, but cannot be absolute or drive-qualified,
cannot contain backslashes, and cannot contain empty, dot, or parent segments.
An invalid value raises `ValueError` during import. These variables set directory defaults during packaging. The writer records
complete bundle-relative paths in bundle.files; adapters then use those paths
for configuration, execution, and evaluation.

For example, set a nested submit layout before starting Python:

~~~console
set DEFAULT_MLFF_TESTING_INPUTS_DIR=inputs/tests
set DEFAULT_MLFF_TESTING_OUTPUTS_DIR=results/tests
~~~

On POSIX shells, use `export` instead of `set`.

## Splitting parameters

| Constant | Type | Default | Environment parsing |
| --- | --- | --- | --- |
| `DEFAULT_TEST_RATIO` | `float` | `0.2` | Parsed with `float`; unset or blank uses the default. |
| `DEFAULT_TRAIN_RATIOS` | `list[float]` | `[0.1, 0.2, 0.4, 0.6, 0.8, 0.9]` | Comma-separated floats; unset or blank uses the default. |
| `DEFAULT_MAX_N_TRAIN` | `int` | `3000` | Parsed with `int`; unset or blank uses the default. |
| `DEFAULT_SPLIT_REPEATS` | `int` | `3` | Parsed with `int`; unset or blank uses the default. |

Invalid non-empty numeric values raise `ValueError` during import. Environment
parsing establishes types only; domain-specific range checks happen where the
values are consumed.

MLFF directory defaults are captured in `bundle.json` when packaging. Remote
runtimes use that stored layout rather than their own environment defaults.
