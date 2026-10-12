"""Shared backend configuration and commands for the installed runtime."""
from __future__ import annotations

from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
import logging
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


logger = logging.getLogger(__name__)


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
    pretrained_model_filename : str
        Default name of the copied model under pretrained_models/. The source
        file remains recorded in MLFFSpec.pretrained_model.artifacts.
    training_input_filenames : dict[str, str]
        Default config and script names relative to training_inputs/.
    training_output_filenames : dict[str, str]
        Default output names relative to training_outputs/, including
        trained_model and training_log. Entries may also name directories.
    evaluation_model_filename : str or None
        Default temporary export name relative to training_outputs/.
        None means the backend does not need a separate evaluation export.
    registry : dict[str, type[BaseMLFFAdapter]]
        Adapter classes indexed by family names and optional aliases.

    The writer turns these defaults into bundle-relative paths in bundle.files.
    During execution, adapters use those recorded paths, not these defaults.
    """

    registry: ClassVar[dict[str, type[BaseMLFFAdapter]]] = {}

    @classmethod
    def register(cls, name: str, alias: str | list[str] | None = None):
        """Return a decorator that registers a runtime adapter.

        Parameters
        ----------
        name : str
            Primary key accepted by mlff_adapter_factory.
        alias : str, list[str], or None, optional
            Additional key or keys referring to the same adapter class.

        Returns
        -------
        Callable
            Decorator that records and returns the supplied class unchanged.
        """
        def decorate(subclass):
            cls.registry[name] = subclass
            if alias is not None:
                for key in ([alias] if isinstance(alias, str) else alias):
                    cls.registry[key] = subclass
            return subclass

        return decorate

    pretrained_model_filename: str
    evaluation_model_filename: str | None = None
    training_input_filenames: ClassVar[dict[str, str]] = {"script": "run.sh"}
    training_output_filenames: ClassVar[dict[str, str]]

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
        """Return the fine-tuned model path recorded in training_output_files.

        The path is relative to the bundle root. Only fine-tuning bundles have
        this entry; zero-shot evaluation uses files.pretrained_model instead.
        """
        return self.files.training_output_files["trained_model"]

    def input_path(self, key: str) -> str:
        """Return a recorded training input path, relative to the bundle root."""
        return self.files.training_input_files[key]

    def output_path(self, key: str) -> str:
        """Return a recorded training output path, relative to the bundle root."""
        return self.files.training_output_files[key]

    @abstractmethod
    def _training_file_contents(self, training_stress: bool) -> dict[str, str]:
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
    def _training_commands(self, training_stress: bool) -> tuple[str, ...]:
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

    def generate_training_files(self, training_stress: bool) -> None:
        """Write native configuration files below the bundle root.

        training_stress enables stress losses when the data contain labels.
        This prepares inspectable inputs without importing training backends.
        """
        for relative, content in self._training_file_contents(training_stress).items():
            self._write_text(relative, content)

    def _write_text(self, relative: str, content: str) -> None:
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8", newline="\n")
        logger.debug("Wrote training input %s", destination)

    def generate_training_script(self, training_stress: bool) -> None:
        """Write the recorded training script and send backend output to its log.

        The script and log use their recorded bundle-relative paths.
        training_stress specifies whether the training set has stress labels.
        Python API backends supply no shell commands and write no script.
        """
        lines = self._training_commands(training_stress)
        if lines:
            self._write_text(
                self.input_path("script"),
                "#!/usr/bin/env bash\nset -euo pipefail\n{\n"
                + "\n".join(lines)
                + "\n} > "
                + shlex.quote(self.output_path("training_log")) + " 2>&1\n",
            )

    def prepare_training(self) -> None:
        """Prepare backend inputs after native configs have been written.

        The runner calls this before training. Adapters may convert datasets
        or resolve checkpoint settings; the default needs no extra preparation.
        No model is trained and no value is returned.
        """

    def prepare_evaluation(self) -> None:
        """Prepare any host-specific model export required for evaluation."""

    def cleanup_evaluation(self) -> None:
        """Remove the temporary evaluation model, including after failed runs."""
        if self.files.evaluation_model is not None:
            temporary = self.root / self.files.evaluation_model
            temporary.unlink(missing_ok=True)
            logger.debug("Removed temporary evaluation model %s", temporary)

    def train(self) -> None:
        """Run the prepared native training script and retain its training log.

        Commands run from the bundle root using the current Python interpreter.
        Returns None on success; a failed command raises CalledProcessError.
        Backends with Python training APIs may override this method.
        """
        subprocess.run(
            ["bash", self.input_path("script")],
            cwd=self.root, env={**os.environ, "PYTHON_BIN": sys.executable}, check=True,
        )
