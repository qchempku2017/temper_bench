"""Shared backend configuration and commands for the installed runtime."""
from __future__ import annotations

from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
import shlex
import os
import subprocess
import sys
from typing import Any, ClassVar, TYPE_CHECKING
from ruamel.yaml import YAML

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
    """Quote a command while preserving the runtime-owned Python shell variable."""
    return " ".join(
        (
            '"$PYTHON_BIN"'
            if argument == "$PYTHON_BIN"
            else shlex.quote(str(argument))
        )
        for argument in arguments
    )


class BaseMLFFAdapter(ABC):
    """Translate a packaged recipe into one backend's native training inputs.

    Parameters
    ----------
    bundle : MLFFTrainBundle
        Loaded bundle with a populated files mapping.
    root : Path
        Absolute directory containing bundle.json and its copied inputs.

    Attributes
    ----------
    files : MLFFBundleFiles
        Input paths, output locations, and label availability from the bundle.
    mlff_type : str
        Registry key for the model family.
    calculator_module : str
        Import path of the family's ASE calculator constructor.
    model_filenames : dict[str, str]
        Artifact keys mapped to filenames used by the local writer.
    trained_model_filename : str
        Default output filename recorded by the writer in the file mapping.
    registry : dict[str, type[BaseMLFFAdapter]]
        Adapter classes indexed by family names and optional aliases.
    """

    registry: ClassVar[dict[str, type[BaseMLFFAdapter]]] = {}

    @classmethod
    def register(cls, name: str, alias: str | None = None):
        """Return a decorator that registers a runtime adapter.

        Parameters
        ----------
        name : str
            Primary key accepted by mlff_adapter_factory.
        alias : str or None, optional
            Additional key referring to the same adapter class.

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
    calculator_module: str
    model_filenames: dict[str, str]
    trained_model_filename: str

    def __init__(self, bundle: MLFFTrainBundle, root: Path) -> None:
        self.bundle = bundle
        self.root = root
        if bundle.files is None:
            raise ValueError("Bundle has not been packaged; call write_submit_folder first.")
        self.files = bundle.files

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
        return f"{self.files.artifacts_dir}/{self.files.trained_model_filename}"

    def artifact_path(self, key: str) -> str:
        """Return the bundle-relative pretrained artifact path for key."""
        return self.files.models[key]

    def dataset_path(self, filename: str) -> str:
        """Map an original TrainingUnit filename to its bundle-relative path."""
        return self.files.datasets[filename]

    @abstractmethod
    def generated_training_files(self, training_stress: bool) -> dict[str, str]:
        """Render native configs without reading checkpoints or training models.

        Parameters
        ----------
        training_stress : bool
            Whether the training data contain stress labels.

        Returns
        -------
        dict[str, str]
            Bundle-relative filenames mapped to their complete text contents.
        """
        return {}

    @abstractmethod
    def training_lines(self, training_stress: bool) -> tuple[str, ...]:
        """Describe native training commands after prepare_training completes.

        Parameters
        ----------
        training_stress : bool
            Whether to enable native stress/virial training controls.

        Returns
        -------
        tuple[str, ...]
            Bash command lines run from the bundle root. Commands must leave
            the trained model at trained_model_path for subsequent evaluation.
        """
        raise NotImplementedError

    def prepare_training(self) -> None:
        """Prepare backend inputs after native configs have been written.

        The runner calls this before training. Adapters may convert datasets
        or resolve checkpoint settings; the default needs no extra preparation.
        No model is trained and no value is returned.
        """

    def train(self) -> None:
        """Run the prepared native training script and retain its training log.

        Commands run from the bundle root using the current Python interpreter.
        Returns None on success; a failed command raises CalledProcessError.
        Backends with Python training APIs may override this method.
        """
        subprocess.run(
            ["bash", f"{self.files.training_dir}/run.sh"],
            cwd=self.root, env={**os.environ, "PYTHON_BIN": sys.executable}, check=True,
        )
