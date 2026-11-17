"""Load predictions and original split-data labels for local analysis.

The public manifest_io module loads bundle metadata and checks it against manifests.
The result_loading module composes MLFFExperimentResult objects when requested.
Users must retain their original
split datasets unchanged; this package does not archive or copy them.
"""

from temper.results_loading import manifest_io, result_loading

__all__ = ["manifest_io", "result_loading"]
