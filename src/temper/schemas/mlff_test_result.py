"""Serializable metadata for one MLFF test dataset.

This standard-library schema is also copied into standalone submit folders,
so result writing does not require TEMPER to be installed on the runner.
Prediction arrays remain in the adjacent NPZ file.
"""
from dataclasses import asdict, dataclass


@dataclass
class MLFFTestResult:
    """Describe one test dataset's predictions and evaluation provenance.

    Metadata is stored as JSON beside the prediction NPZ file. This schema
    uses only the standard library so it can be copied into a standalone
    submit folder and used without a TEMPER installation.

    Parameters
    ----------
    schema_version : int
        Version of the JSON metadata contract.
    calculator_identifier : str
        MLFF family used to construct the evaluation calculator.
    model : str
        Evaluated model path relative to the submit folder.
    package_versions : dict[str, str or None]
        Installed distribution versions; None denotes an unavailable version.
    dataset_id : str
        Dataset identifier shared by the JSON and NPZ output filenames.
    source_domain : str
        Original data domain from the training unit.
    source_filename : str
        Original extxyz path relative to the source domain.
    submit_path : str
        Copied extxyz path relative to the submit folder.
    number_of_frames : int
        Number of evaluated structures.
    total_atoms : int
        Sum of atom counts across all evaluated structures.
    requested_properties : list[str]
        Physical quantities requested from the calculator.
    units : dict[str, str or None]
        Units for each output quantity; None marks an omitted quantity.
    wall_time_seconds : float
        Elapsed time for evaluation and writing the prediction arrays.
    """

    schema_version: int
    calculator_identifier: str
    model: str
    package_versions: dict[str, str | None]
    dataset_id: str
    source_domain: str
    source_filename: str
    submit_path: str
    number_of_frames: int
    total_atoms: int
    requested_properties: list[str]
    units: dict[str, str | None]
    wall_time_seconds: float

    def as_dict(self) -> dict:
        """Return plain JSON-compatible metadata."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "MLFFTestResult":
        """Return a result loaded from its JSON metadata dictionary."""
        return cls(**value)
