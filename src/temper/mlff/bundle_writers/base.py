"""Shared mechanics for fixed-layout MLFF submit folders."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import tempfile
from copy import deepcopy
from importlib import resources
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import iread
from ruamel.yaml import YAML

from temper.schemas.mlff_spec import LocalArtifactRef
from temper.schemas.utils import (
    validate_relative_extxyz_path,
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


def yaml_text(value: dict[str, Any]) -> str:
    """Serialize one native package configuration as readable YAML."""
    yaml = YAML()
    yaml.default_flow_style = False
    stream = StringIO()
    yaml.dump(value, stream)
    return stream.getvalue()


def command(*arguments: Any) -> str:
    """Quote a command while preserving the two writer-owned shell variables."""
    return " ".join(
        (
            '"$PYTHON_BIN"'
            if argument == "$PYTHON_BIN"
            else '"$MLFF_DEVICE"'
            if argument == "$MLFF_DEVICE"
            else shlex.quote(str(argument))
        )
        for argument in arguments
    )


class BaseMLFFBundleWriter:
    """Copy common inputs and let a concrete writer add native training files."""

    mlff_type: str
    calculator_resource: str
    model_filenames: dict[str, str]
    trained_model_filename: str

    def __init__(self, bundle: MLFFTrainBundle) -> None:
        self.bundle = bundle

    @property
    def spec(self):
        """Return the nested MLFF specification."""
        return self.bundle.mlff_spec

    @property
    def unit(self):
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

    def _dataset_source(self, filename: str) -> Path:
        validate_relative_extxyz_path(filename)
        domain = Path(self.unit.domain)
        if (
            domain.is_absolute()
            or domain.root
            or len(domain.parts) != 1
            or ".." in domain.parts
        ):
            raise ValueError("TrainingUnit domain must be one safe directory name.")
        domain_root = (self.unit.root_path / domain).resolve()
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
        has_stress: bool | None = None
        frame_count = 0
        for frame_index, atoms in enumerate(iread(source, index=":")):
            frame_count += 1
            try:
                atoms.get_potential_energy()
            except Exception as error:
                raise ValueError(
                    f"{source} frame {frame_index} is missing energy."
                ) from error
            try:
                atoms.get_forces()
            except Exception as error:
                raise ValueError(
                    f"{source} frame {frame_index} is missing forces."
                ) from error
            try:
                atoms.get_stress()
                frame_has_stress = True
            except PropertyNotImplementedError:
                frame_has_stress = False
            except Exception as error:
                raise ValueError(
                    f"{source} frame {frame_index} has an invalid stress label."
                ) from error
            if has_stress is not None and frame_has_stress != has_stress:
                raise ValueError(
                    f"{source} mixes frames with and without stress labels."
                )
            has_stress = frame_has_stress
        if frame_count == 0:
            raise ValueError(f"MLFF dataset is empty: {source}.")
        return bool(has_stress)

    def inspect_datasets(self) -> tuple[bool | None, list[bool]]:
        """Validate labels and return training and per-test stress availability."""
        training_stress: bool | None = None
        if self.bundle.unit_type == "finetune":
            assert self.unit.train_set is not None
            training_stress = self._dataset_has_stress(
                self._dataset_source(self.unit.train_set)
            )
            if self.unit.val_set is not None:
                validation_stress = self._dataset_has_stress(
                    self._dataset_source(self.unit.val_set)
                )
                if validation_stress != training_stress:
                    raise ValueError(
                        "Training and validation datasets disagree on stress "
                        "availability."
                    )
        test_stress = [
            self._dataset_has_stress(self._dataset_source(filename))
            for filename in self.unit.test_sets
        ]
        return training_stress, test_stress

    def validate_bundle(self) -> None:
        """Apply the few invariants shared by every concrete writer."""
        if self.spec.mlff_type != self.mlff_type:
            raise ValueError(
                f"{type(self).__name__} cannot write {self.spec.mlff_type!r}."
            )
        expected = set(self.model_filenames)
        actual = set(self.spec.pretrained_model.artifacts)
        if actual != expected:
            raise ValueError(
                f"{self.mlff_type} pretrained artifacts must be "
                f"{sorted(expected)!r}; got {sorted(actual)!r}."
            )

    def generated_training_files(self, training_stress: bool) -> dict[str, str]:
        """Return package-native configurations keyed by submit-relative path."""
        return {}

    def training_lines(self, training_stress: bool) -> tuple[str, ...]:
        """Return shell lines that fine-tune and place the standardized model."""
        raise NotImplementedError

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
            zip(self.unit.test_sets, test_stress, strict=True)
        ):
            stem = f"test_{index:03d}"
            properties = ["energy", "forces"]
            if has_stress:
                properties.append("stress")
            datasets.append(
                {
                    "id": stem,
                    "path": f"{DEFAULT_MLFF_DATASETS_DIR}/{stem}.extxyz",
                    "source_domain": self.unit.domain,
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
                "parameters": deepcopy(self.spec.testing),
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

    def run_script(self, training_stress: bool) -> str:
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
            **self.extra_runtime_resources(),
        }
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
            assert self.unit.train_set is not None
            self._copy(
                self._dataset_source(self.unit.train_set),
                root,
                f"{DEFAULT_MLFF_DATASETS_DIR}/train.extxyz",
            )
            if self.unit.val_set is not None:
                self._copy(
                    self._dataset_source(self.unit.val_set),
                    root,
                    f"{DEFAULT_MLFF_DATASETS_DIR}/validation.extxyz",
                )
        for index, filename in enumerate(self.unit.test_sets):
            self._copy(
                self._dataset_source(filename),
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
        self.validate_bundle()
        training_stress, test_stress = self.inspect_datasets()
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
                self.run_script(bool(training_stress)),
            )
            (target / "run.sh").chmod(0o755)
        except Exception:
            shutil.rmtree(target)
            raise
        return target
