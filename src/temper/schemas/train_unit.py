"""Defines persisted benchmark data units and their extxyz references."""

from functools import cached_property
import json

from temper.utils.extxyz import check_extxyz_properties, iter_extxyz_metadata
from temper.schemas.utils import validate_relative_extxyz_path

from typing import Any, ClassVar, Literal
from pathlib import Path
from uuid import UUID

from pydantic import (
    Field,
    field_serializer,
    field_validator,
    model_validator,
    ValidationInfo,
)

from temper.schemas.base import ManagedIdentityModel
from temper.utils.defaults import DEFAULT_SPLIT_RESULTS_DIR
from temper.utils.file_hashing import file_sha256, _verification_cache


_TRAINING_UNIT_ID_NAMESPACE = UUID("a219bd97-5b63-5dc1-8543-38d74c746ecf")


class TrainingUnit(ManagedIdentityModel):
    """Persisted datasets for one fine-tuning or zero-shot benchmark.

    ``unit_type`` is derived from dataset shape rather than stored separately:
    a unit with ``train_set`` is ``"finetune"`` and a unit without one is
    ``"zeroshot"``. Fine-tuning units require a non-empty training dataset.
    Zero-shot units contain no training or validation dataset and reuse the
    same evaluation datasets as fine-tuning units from their ``SplitGroup``.

    A unit belongs to:
        - A specific data domain
        - A specific grouping strategy (The level of GroupedDomain)
        - A specific group produced by the grouping strategy.
        - A specific train-val split method.
        - A specific repeat_id among independent train-val-test splits on the group using the method.
            (The level of SplitGroup)
        - Either a specific training-frame checkpoint or the zero-shot
          evaluation derived from one independent train-val-test split.

    The unit contains extxyz file paths containing labeled data:
        - A training set (a single file path, fine-tuning only)
        - A validation set (a single file path, optional and fine-tuning only)
        - A test set (a list of file paths, each corresponding to a tested group)

    Usually produced by src.temper.splitting.io ``write_all_sets_in_split_group_to_extxyz``.
    It describes benchmark input data; it does not train or evaluate an MLFF.

    Fields remain mutable through validated reassignment. The system-managed
    ``training_unit_id`` is stored, verified when loaded, and regenerated when
    an identity-defining field changes. ``root_path`` can be relocated without
    changing the identity.

    Parameters
    ----------
    **data : object
        Values for the attributes below. Normal loading of a record containing
        dataset_sha256 rereads and verifies the referenced bytes. For historical
        inspection, use model_validate or model_validate_json with
        context={"verify_files": False}; this does not authorize execution.

    Attributes
    ----------
    domain : str
        Name of the data domain.
    grouping_strategy : str
        Name of the grouping strategy.
    group_name : str
        Name of the group.
    method : str
        Name of the train-validation split method.
    repeat_id : int
        Nonnegative repeat ID of the independent split.
    unit_type : Literal["finetune", "zeroshot"]
        Read-only mode derived from whether train_set is present.
    train_n_frames, val_n_frames, test_n_frames : int
        Nonnegative frame counts in the training, validation and combined test
        datasets. Training counts are zero for zero-shot units; validation
        counts are zero when no validation file is exported.
    train_n_atoms, val_n_atoms, test_n_atoms : int
        Total atoms across the corresponding frames, with the same absence
        conventions as the frame counts.
    split_id : UUID or None
        Identity of the SplitGroup that produced this unit. None is accepted
        for manifests written before split identities were introduced.
    training_unit_id : UUID or None
        Stored, system-managed identity. None requests identity creation and
        is populated before a valid model is returned. Hashed units use v3
        identity rules; legacy metadata retains its v2 identity until upgraded.
    train_set : str or None
        Training filename, relative to root_path/domain; None for zero-shot.
    test_sets : tuple[str, ...]
        Test filenames, relative to root_path/domain, in evaluation order.
    val_set : str or None
        Validation filename, relative to root_path/domain; None when absent.
    root_path : Path
        Movable dataset root, default DEFAULT_SPLIT_RESULTS_DIR. Relative roots
        resolve against the current working directory, not a JSON file's parent.
        Files are read from root_path/domain/filename.
    dataset_sha256 : dict[str, str]
        Dataset filenames mapped to hashes of the complete file bytes. Data
        splitting populates this before returning each unit. Empty means legacy
        metadata: inspection and local packaging remain supported, but remote
        submission requires explicit with_dataset_hashes() conversion.

    Notes
    -----
    Changing a source file does not mutate an already loaded Python object's
    identity. Loading or submitting its hashed record detects the mismatch.
    with_dataset_hashes() explicitly creates a new identity from changed files.
    Hash caches live only for one creation, packaging or submission operation;
    file size and modification time are never used as proof of unchanged bytes.
    """
    _IDENTITY_FIELD_NAME: ClassVar[str] = "training_unit_id"
    _IDENTITY_SOURCE_FIELDS: ClassVar[tuple[str, ...]] = (
        "split_id",
        "domain",
        "grouping_strategy",
        "group_name",
        "method",
        "repeat_id",
        "train_n_frames",
        "train_set",
        "test_sets",
        "val_set",
        "dataset_sha256",
    )
    _IDENTITY_NAMESPACE: ClassVar[UUID] = _TRAINING_UNIT_ID_NAMESPACE
    _IDENTITY_SCHEMA: ClassVar[str] = "temper.training-unit.v2"
    _IDENTITY_LABEL: ClassVar[str] = "training-unit"

    domain: str
    grouping_strategy: str
    group_name: str
    method: str

    repeat_id: int = Field(
        ge=0,
    )

    train_n_frames: int = Field(
        ge=0,
    )
    val_n_frames: int = Field(
        ge=0,
    )
    test_n_frames: int = Field(
        ge=0,
    )

    train_n_atoms: int = Field(
        ge=0,
    )
    val_n_atoms: int = Field(
        ge=0,
    )
    test_n_atoms: int = Field(
        ge=0,
    )

    split_id: UUID | None = None

    train_set: str | None

    test_sets: tuple[str, ...]

    val_set: str | None = None

    root_path: Path = Field(
        default=DEFAULT_SPLIT_RESULTS_DIR,
        validate_default=True,
    )
    training_unit_id: UUID | None = None
    dataset_sha256: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _verify_dataset_bytes(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, dict):
            return data
        hashes = data.get("dataset_sha256", {})
        if not hashes:
            return data  # Legacy metadata remains inspectable, but cannot be submitted.
        names = {name for name in (*data.get("test_sets", ()), data.get("train_set"), data.get("val_set")) if name}
        if set(hashes) != names:
            raise ValueError("dataset_sha256 must cover exactly the training, validation and test files.")
        if any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes.values()):
            raise ValueError("Dataset digests must be lowercase SHA-256 strings.")
        context = info.context or {}
        if context.get("verify_files", True):
            cache = context.get("hash_cache")
            if cache is None:
                cache = _verification_cache.get()
            if cache is None:
                cache = {}
            for name, expected in hashes.items():
                validate_relative_extxyz_path(name)
                root = (Path(data.get("root_path", DEFAULT_SPLIT_RESULTS_DIR)) / data["domain"]).resolve()
                path = (root / name).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Dataset escapes its domain directory.")
                if path not in cache:
                    cache[path] = file_sha256(path)
                if cache[path] != expected:
                    raise ValueError(f"Dataset bytes changed: {path}. Rebuild the training unit explicitly.")
        return data

    def _identity_payload(self) -> dict[str, Any]:
        payload = super()._identity_payload()
        payload.pop("dataset_sha256", None)
        if self.dataset_sha256:
            payload["identity_schema"] = "temper.training-unit.v3"
            payload["dataset_sha256"] = self.dataset_sha256
        return payload

    def with_dataset_hashes(self, cache: dict[Path, str] | None = None) -> "TrainingUnit":
        """Create a new identity from the current dataset bytes.

        Parameters
        ----------
        cache : dict[Path, str] or None, optional
            Digests shared only within this creation operation. None reads each
            distinct file once. Never reuse a cache across separate operations.

        Returns
        -------
        TrainingUnit
            Independent record with hashes for all datasets and a new identity.
            This is also the explicit upgrade operation for legacy metadata.

        Raises
        ------
        OSError
            If a dataset cannot be read.
        ValueError
            If a reference is invalid or missing.
        """
        cache = {} if cache is None else cache
        hashes: dict[str, str] = {}
        for name in (*self.test_sets, self.train_set, self.val_set):
            if name is not None:
                path = self.dataset_source(name)
                if path not in cache:
                    cache[path] = file_sha256(path)
                hashes[name] = cache[path]
        data = self.model_dump(exclude={"training_unit_id"})
        data["dataset_sha256"] = hashes
        return type(self).model_validate(data, context={"hash_cache": cache})

    def dataset_source(self, filename: str) -> Path:
        """Return the existing dataset path within this unit\'s domain."""
        validate_relative_extxyz_path(filename)
        domain = Path(self.domain)
        if (
            domain.is_absolute()
            or domain.root
            or len(domain.parts) != 1
            or ".." in domain.parts
        ):
            raise ValueError("TrainingUnit domain must be one safe directory name.")
        domain_root = (self.root_path / domain).resolve()
        source = (domain_root / filename).resolve()
        try:
            source.relative_to(domain_root)
        except ValueError as error:
            raise ValueError(
                f"TrainingUnit dataset escapes its domain root: {filename!r}."
            ) from error
        if not source.is_file():
            raise ValueError(f"TrainingUnit dataset does not exist: {source}.")
        return source

    @staticmethod
    def _dataset_has_stress(source: Path) -> bool:
        """Return whether every frame has stress; require energy and forces."""
        has_stress: bool | None = None
        frame_count = 0
        for frame_index, frame in enumerate(iter_extxyz_metadata(source)):
            frame_count += 1
            frame_has_stress = check_extxyz_properties(
                frame, source=f"{source} frame {frame_index}",
            )
            if has_stress is not None and frame_has_stress != has_stress:
                raise ValueError(
                    f"{source} mixes frames with and without stress labels."
                )
            has_stress = frame_has_stress
        if frame_count == 0:
            raise ValueError(f"MLFF dataset is empty: {source}.")
        return bool(has_stress)


    def __setattr__(self, name: str, value: Any) -> None:
        """Invalidate inspected labels when dataset references change."""
        super().__setattr__(name, value)
        if name in {"root_path", "domain", "train_set", "val_set", "test_sets"}:
            self.__dict__.pop("dataset_stress", None)

    @cached_property
    def dataset_stress(self) -> dict[str, bool]:
        """Return cached stress availability by dataset filename.

        Each dataset must have energy and forces on every frame and consistent
        stress labels. Exported datasets are treated as immutable after inspection.
        """
        filenames = (*self.test_sets, self.train_set, self.val_set)
        return {
            name: self._dataset_has_stress(self.dataset_source(name))
            for name in dict.fromkeys(filenames) if name is not None
        }

    @property
    def training_has_stress(self) -> bool | None:
        """Return training stress availability, or None for zero-shot units."""
        if self.train_set is None:
            return None
        result = self.dataset_stress[self.train_set]
        if self.val_set and self.dataset_stress[self.val_set] != result:
            raise ValueError("Training and validation datasets disagree on stress availability.")
        return result

    @property
    def test_has_stress(self) -> list[bool]:
        """Return stress availability in the same order as test_sets."""
        return [self.dataset_stress[name] for name in self.test_sets]

    @property
    def unit_type(self) -> Literal["finetune", "zeroshot"]:
        """Return the benchmark mode implied by the training-set shape."""
        return "finetune" if self.train_set is not None else "zeroshot"

    @field_serializer("root_path")
    def serialize_root_path(self, value: Path) -> str:
        """Serialize the movable root as a portable path string."""
        return str(value)

    @field_serializer("split_id")
    def serialize_split_id(self, value: UUID | None) -> str | None:
        """Serialize the parent identity as a standard UUID string."""
        return None if value is None else str(value)

    @field_validator("root_path", mode="before")
    @classmethod
    def load_monty_root_path(cls, value: Any) -> Any:
        """Accept path dictionaries written by earlier Monty encoders."""
        if isinstance(value, dict) and value.get("@module") == "pathlib":
            return value.get("string", value)
        return value

    @field_validator("split_id", mode="before")
    @classmethod
    def load_monty_split_id(cls, value: Any) -> Any:
        """Accept UUID dictionaries written by Monty encoders."""
        if isinstance(value, dict) and value.get("@module") == "uuid":
            return value.get("string", value)
        return value

    def _validate_before_identity(self) -> None:
        """Validate dataset shape and names without accessing local source files.

        Records also travel inside remote bundles. File existence is checked
        by dataset_source when local inputs are actually used.
        """

        if self.unit_type == "finetune":
            if self.train_n_frames <= 0:
                raise ValueError(
                    "Fine-tuning TrainingUnit requires train_n_frames > 0."
                )
            if self.train_n_atoms <= 0:
                raise ValueError(
                    "Fine-tuning TrainingUnit requires train_n_atoms > 0."
                )
        else:
            if self.train_n_frames != 0:
                raise ValueError(
                    "Zero-shot TrainingUnit requires train_n_frames == 0."
                )
            if self.train_n_atoms != 0:
                raise ValueError(
                    "Zero-shot TrainingUnit requires train_n_atoms == 0."
                )
            if self.val_set is not None:
                raise ValueError(
                    "Zero-shot TrainingUnit cannot reference a validation set."
                )
            if self.val_n_frames != 0:
                raise ValueError(
                    "Zero-shot TrainingUnit requires val_n_frames == 0."
                )
            if self.val_n_atoms != 0:
                raise ValueError(
                    "Zero-shot TrainingUnit requires val_n_atoms == 0."
                )

        def _check_extxyz_file(f: str) -> None:
            file_path = self.root_path / self.domain / f

            if file_path.suffix != ".extxyz":
                raise ValueError(
                    f"Dataset file must have .extxyz extension, got: "
                    f"{f}."
                )

        if self.train_set is not None:
            _check_extxyz_file(self.train_set)

        if self.val_set is not None:
            _check_extxyz_file(self.val_set)

        for filename in self.test_sets:
            _check_extxyz_file(filename)


def load_training_units(path: str | Path, *, verify_files: bool = True) -> list[TrainingUnit]:
    """Load a JSON list of units, hashing shared datasets once in this operation.

    Parameters
    ----------
    path : str or Path
        JSON manifest path, relative to the current working directory. Dataset
        root_path values retain their own current-directory interpretation.
    verify_files : bool, default True
        Reread and verify the bytes recorded in hashed units. False permits
        historical inspection when source files are unavailable; remote
        submission still verifies those files separately. Legacy units remain
        metadata-only until explicitly converted with with_dataset_hashes().

    Returns
    -------
    list[TrainingUnit]
        Validated records in file order. No persistent size/time cache is used.

    Raises
    ------
    ValueError
        If the file is not a JSON list, an identity is invalid, or data changed.
    OSError
        If the manifest or a required dataset cannot be read.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Training-unit manifest must contain a JSON list.")
    context = {"verify_files": verify_files, "hash_cache": {}}
    return [TrainingUnit.model_validate(item, context=context) for item in data]
