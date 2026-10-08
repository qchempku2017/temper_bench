"""Shared mechanics for fixed-layout MLFF submit folders."""

from __future__ import annotations

from abc import ABC, abstractmethod

import hashlib
import json
import shlex
import shutil
import tempfile
from copy import deepcopy
from importlib import resources
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar

from ruamel.yaml import YAML

from temper.schemas.artifact import LocalArtifactRef
from temper.schemas.utils import (
    validate_submit_relative_path,
)
from temper.utils.defaults import (
    DEFAULT_MLFF_ARTIFACTS_DIR,
    DEFAULT_MLFF_DATASETS_DIR,
    DEFAULT_MLFF_MODELS_DIR,
    DEFAULT_MLFF_OUTPUTS_DIR,
    DEFAULT_MLFF_RUNTIME_DIR,
)

if TYPE_CHECKING:
    from temper.schemas.mlff_train_bundle import MLFFTrainBundle
    from temper.schemas.mlff_spec import MLFFSpec
    from temper.schemas.train_unit import TrainingUnit


def yaml_text(value: dict[str, Any]) -> str:
    """Serialize one native package configuration as readable YAML."""
    yaml = YAML()
    yaml.default_flow_style = False
    stream = StringIO()
    yaml.dump(value, stream)
    return stream.getvalue()


def command(*arguments: Any) -> str:
    """Quote a command while preserving the writer-owned Python shell variable."""
    return " ".join(
        (
            '"$PYTHON_BIN"'
            if argument == "$PYTHON_BIN"
            else shlex.quote(str(argument))
        )
        for argument in arguments
    )


class BaseMLFFBundleWriter(ABC):
    """Write one atomic bundle using a fixed portable directory layout.

    Attributes
    ----------
    bundle : MLFFTrainBundle
        Pair of a dataset unit and its reproducible MLFF recipe.
    mlff_type : str
        Family key used to register the concrete writer.
    calculator_resource : str
        Packaged runtime adapter path, relative to temper.mlff.runtime.
    model_filenames : dict[str, str]
        Artifact keys mapped to the filenames to be written within the submit
        models directory. (Not the artifact's original filename before copying).
    trained_model_filename : str
        Trained model filename within the submit artifacts directory.
    registry : dict
        Concrete writers keyed by MLFF family.
    """

    registry: ClassVar[dict[str, type[BaseMLFFBundleWriter]]] = {}

    @classmethod
    def register(cls, name: str, alias: str | None = None):
        """Return a decorator that registers a submit-folder writer.

        Parameters
        ----------
        name : str
            Primary key accepted by mlff_bundle_writer_factory.
        alias : str or None, optional
            Additional key referring to the same writer class.

        Returns
        -------
        Callable
            Decorator that records and returns the supplied class unchanged.
        """
        def decorate(subclass):
            cls.registry[name] = subclass
            if alias is not None:
                cls.registry[alias] = subclass
            return subclass

        return decorate

    mlff_type: str
    calculator_resource: str
    model_filenames: dict[str, str]
    trained_model_filename: str

    def __init__(self, bundle: MLFFTrainBundle) -> None:
        self.bundle = bundle

    @property
    def spec(self) -> MLFFSpec:
        """Return the nested MLFF specification."""
        return self.bundle.mlff_spec

    @property
    def training_unit(self) -> TrainingUnit:
        """Return the nested benchmark data unit."""
        return self.bundle.training_unit

    @property
    def trained_model_path(self) -> str:
        """Return the fixed submit path for this family's trained model."""
        return f"{DEFAULT_MLFF_ARTIFACTS_DIR}/{self.trained_model_filename}"

    def artifact_path(self, key: str) -> str:
        """Return the fixed submit path for one pretrained artifact key."""
        try:
            filename = self.model_filenames[key]
        except KeyError as error:
            raise ValueError(
                f"{self.mlff_type} does not use pretrained artifact {key!r}."
            ) from error
        return f"{DEFAULT_MLFF_MODELS_DIR}/{filename}"

    def artifact(self, key: str) -> LocalArtifactRef:
        """Return one required artifact or fail with a concise schema error."""
        try:
            return self.spec.pretrained_model.artifacts[key]
        except KeyError as error:
            raise ValueError(
                f"{self.mlff_type} requires pretrained artifact {key!r}."
            ) from error

    @staticmethod
    def _verify_artifact(artifact: LocalArtifactRef) -> Path:
        path = artifact.path.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Local MLFF artifact does not exist: {path}.")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != artifact.sha256:
            raise ValueError(
                "Local MLFF artifact changed after the specification was built: "
                f"{path}; expected {artifact.sha256}, got {actual}."
            )
        return path

    @abstractmethod
    def generated_training_files(self, training_stress: bool) -> dict[str, str]:
        """Return package-native configurations keyed by submit-relative path."""
        return {}

    @abstractmethod
    def training_lines(self, training_stress: bool) -> tuple[str, ...]:
        """Return shell lines that fine-tune and place the standardized model."""
        raise NotImplementedError

    @abstractmethod
    def extra_runtime_resources(self) -> dict[str, str]:
        """Map extra runtime destination names to packaged resource paths."""
        return {}

    def test_config(self, test_stress: list[bool]) -> dict[str, Any]:
        """Build the common evaluation configuration with per-dataset properties."""
        model_path = (
            self.trained_model_path
            if self.bundle.unit_type == "finetune"
            else self.artifact_path("model")
        )
        datasets = []
        for index, (filename, has_stress) in enumerate(
            zip(self.training_unit.test_sets, test_stress, strict=True)
        ):
            stem = f"test_{index:03d}"
            properties = ["energy", "forces"]
            if has_stress:
                properties.append("stress")
            datasets.append(
                {
                    "id": stem,
                    "path": f"{DEFAULT_MLFF_DATASETS_DIR}/{stem}.extxyz",
                    "source_domain": self.training_unit.domain,
                    "source_filename": filename,
                    "properties": properties,
                    "output": f"{DEFAULT_MLFF_OUTPUTS_DIR}/{stem}.npz",
                    "metadata_output": f"{DEFAULT_MLFF_OUTPUTS_DIR}/{stem}.json",
                }
            )
        return {
            "schema_version": 2,
            "calculator": {
                "identifier": self.mlff_type,
                "parameters": deepcopy(self.spec.testing_parameters),
            },
            "model": model_path,
            "test_datasets": datasets,
            "summary_output": f"{DEFAULT_MLFF_OUTPUTS_DIR}/test_summary.json",
            "package_requirements": [
                {"name": item.name, "version": item.version}
                for item in self.spec.implementations
                if item.kind == "python_distribution"
            ],
        }

    def generate_run_script(self, training_stress: bool) -> str:
        """Render the fixed entry script and any package-native training stage."""
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'cd -- "$(dirname -- "$0")"',
            'PYTHON_BIN="' + "$" + '{PYTHON:-python}"',
            command(
                "mkdir",
                "-p",
                DEFAULT_MLFF_ARTIFACTS_DIR,
                DEFAULT_MLFF_OUTPUTS_DIR,
            ),
        ]
        preflight = ["$PYTHON_BIN", f"{DEFAULT_MLFF_RUNTIME_DIR}/check_cuda.py"]
        if self.mlff_type == "mattersim" and self.bundle.unit_type == "finetune":
            preflight.append("--warn-mattersim")
        lines.append(command(*preflight))
        if self.bundle.unit_type == "finetune":
            lines.append("{")
            lines.extend(f"  {line}" for line in self.training_lines(training_stress))
            lines.append(
                "} 2>&1 | tee "
                + shlex.quote(f"{DEFAULT_MLFF_OUTPUTS_DIR}/training.log")
            )
        lines.append(
            command(
                "$PYTHON_BIN",
                f"{DEFAULT_MLFF_RUNTIME_DIR}/run_test.py",
                "--config",
                "test_config.json",
            )
        )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _destination(root: Path, relative: str) -> Path:
        validate_submit_relative_path(relative, label="submit path")
        return root.joinpath(*PurePosixPath(relative).parts)

    def _copy(self, source: Path, root: Path, relative: str) -> None:
        destination = self._destination(root, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def _write_text(self, root: Path, relative: str, content: str) -> None:
        destination = self._destination(root, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8", newline="\n")

    def _copy_runtime(self, root: Path) -> None:
        packaged = resources.files("temper.mlff.runtime")
        selected = {
            "run_test.py": "run_test.py",
            "calculator.py": self.calculator_resource,
            "check_cuda.py": "check_cuda.py",
            **self.extra_runtime_resources(),
        }
        schema = resources.files("temper.schemas").joinpath("mlff_test_result.py")
        with resources.as_file(schema) as source:
            self._copy(source, root, f"{DEFAULT_MLFF_RUNTIME_DIR}/result_schema.py")
        for destination_name, resource_name in selected.items():
            resource = packaged.joinpath(resource_name)
            if not resource.is_file():
                raise FileNotFoundError(f"Runtime resource is missing: {resource_name}.")
            destination = self._destination(
                root, f"{DEFAULT_MLFF_RUNTIME_DIR}/{destination_name}"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with resources.as_file(resource) as source:
                shutil.copy2(source, destination)

    def _copy_inputs(self, root: Path) -> None:
        if self.bundle.unit_type == "finetune":
            assert self.training_unit.train_set is not None
            self._copy(
                self.training_unit.dataset_source(self.training_unit.train_set),
                root,
                f"{DEFAULT_MLFF_DATASETS_DIR}/train.extxyz",
            )
            if self.training_unit.val_set is not None:
                self._copy(
                    self.training_unit.dataset_source(self.training_unit.val_set),
                    root,
                    f"{DEFAULT_MLFF_DATASETS_DIR}/validation.extxyz",
                )
        for index, filename in enumerate(self.training_unit.test_sets):
            self._copy(
                self.training_unit.dataset_source(filename),
                root,
                f"{DEFAULT_MLFF_DATASETS_DIR}/test_{index:03d}.extxyz",
            )
        for key, filename in self.model_filenames.items():
            self._copy(
                self._verify_artifact(self.artifact(key)),
                root,
                f"{DEFAULT_MLFF_MODELS_DIR}/{filename}",
            )

    def write_submit_folder(
        self, target_dir: str | Path | None = None
    ) -> Path:
        """Create and return one complete submit directory using file copies."""
        training_stress = self.training_unit.training_has_stress
        test_stress = self.training_unit.test_has_stress
        if target_dir is None:
            target = Path(tempfile.mkdtemp(prefix="temper-submit-"))
        else:
            target = Path(target_dir).expanduser().absolute()
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Target submit folder already exists: {target}.")
            target.mkdir(parents=True)

        try:
            self._copy_inputs(target)
            self._copy_runtime(target)
            if self.bundle.unit_type == "finetune":
                assert training_stress is not None
                for path, content in self.generated_training_files(
                    training_stress
                ).items():
                    self._write_text(target, path, content)
            self._write_text(
                target,
                "test_config.json",
                json.dumps(
                    self.test_config(test_stress), indent=2, sort_keys=True
                )
                + "\n",
            )
            self._write_text(
                target,
                "run.sh",
                self.generate_run_script(bool(training_stress)),
            )
            (target / "run.sh").chmod(0o755)
        except Exception:
            shutil.rmtree(target)
            raise
        return target
