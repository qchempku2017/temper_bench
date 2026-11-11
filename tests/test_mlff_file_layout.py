"""Check that saved specifications completely determine native file names."""

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from temper.mlff.pipeline.training_adapters.base import BaseMLFFAdapter
from temper.mlff.spec_builders import MACESpecBuilder, mlff_spec_builder_factory
from temper.mlff.spec_builders.base import BaseSpecBuilder
from temper.schemas import (
    MLFFFileLayout,
    MLFFSpec,
    MLFFTrainBundle,
    TrainingUnit,
)


@pytest.mark.parametrize(
    "family", ["dpa4", "dpa4c", "mace", "mattersim", "sevennet", "nep89"]
)
@pytest.mark.parametrize("finetune", [True, False])
def test_saved_layout_packages_without_builders_or_adapters(
    family: str,
    finetune: bool,
    tmp_path: Path,
    finetune_training_unit: TrainingUnit,
    zeroshot_training_unit: TrainingUnit,
    mlff_spec_factory: Callable[..., MLFFSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package saved custom names without looking up either registry.

    Parameters
    ----------
    family : str
        Supported MLFF family whose complete layout is exercised.
    finetune : bool
        Whether training paths should be included in the packaged record.
    tmp_path : Path
        Temporary parent directory for packaged inputs.
    finetune_training_unit, zeroshot_training_unit : TrainingUnit
        Source units with readable datasets for the selected mode.
    mlff_spec_factory : Callable[..., MLFFSpec]
        Fixture constructing specifications and local checkpoints.
    monkeypatch : pytest.MonkeyPatch
        Fixture removing registries and changing installed defaults.
    """
    spec = mlff_spec_factory(family, with_training=finetune)
    layout = spec.file_layout
    layout.pretrained_model_filename = (
        "custom/" + layout.pretrained_model_filename
    )
    for mapping in (
        layout.training_input_filenames, layout.training_output_filenames
    ):
        for key, name in mapping.items():
            mapping[key] = "custom/" + name
    if layout.evaluation_model_filename is not None:
        layout.evaluation_model_filename = (
            "custom/" + layout.evaluation_model_filename
        )
    bundle = MLFFTrainBundle(
        mlff_spec=spec,
        training_unit=(
            finetune_training_unit if finetune else zeroshot_training_unit
        ),
    )
    saved = bundle.model_dump_json()
    builder = mlff_spec_builder_factory(family)
    monkeypatch.setattr(builder, "default_file_layout", None)
    monkeypatch.setattr(BaseSpecBuilder, "registry", {})
    monkeypatch.setattr(BaseMLFFAdapter, "registry", {})

    restored = MLFFTrainBundle.model_validate_json(saved)
    target, packaged = restored.write_submit_folder(tmp_path / "job")
    assert restored.files is None
    assert packaged.mlff_spec.file_layout == layout
    files = packaged.files
    assert files is not None
    assert files.pretrained_model == (
        "pretrained_models/" + layout.pretrained_model_filename
    )
    assert (target / files.pretrained_model).read_bytes() == (
        spec.pretrained_model.artifacts["model"].path.read_bytes()
    )
    assert files.training_output_files == (
        {
            key: "training_outputs/" + name
            for key, name in layout.training_output_filenames.items()
        }
        if finetune
        else {}
    )
    if finetune:
        for key, name in layout.training_input_filenames.items():
            assert files.training_input_files[key] == "training_inputs/" + name
    else:
        assert files.training_input_files == {}
    assert files.evaluation_model == (
        "training_outputs/" + layout.evaluation_model_filename
        if finetune and layout.evaluation_model_filename is not None
        else None
    )
    assert MLFFTrainBundle.model_validate_json(
        (target / "bundle.json").read_text()
    ) == packaged


def test_builder_layout_copies_and_scientific_identity(tmp_path: Path) -> None:
    """Isolate mutable layouts while keeping filenames out of scientific IDs.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory containing the source checkpoint.
    """
    (tmp_path / "source.model").write_bytes(b"weights")
    custom = MACESpecBuilder.default_file_layout.model_copy(deep=True)
    custom.training_input_filenames["config"] = "custom/mace.yaml"
    builder = MACESpecBuilder(
        model_name="test",
        model_version="1",
        pretrained_model_dir=tmp_path,
        model_filename="source.model",
        training_parameters={},
        file_layout=custom,
    )
    first, second = builder.build(), builder.build()
    custom.training_input_filenames["config"] = "caller.yaml"
    first.file_layout.training_input_filenames["config"] = "first.yaml"
    first.file_layout.pretrained_model_filename = "renamed.model"
    assert (
        second.file_layout.training_input_filenames["config"]
        == "custom/mace.yaml"
    )
    assert (
        builder.file_layout.training_input_filenames["config"]
        == "custom/mace.yaml"
    )
    assert (
        MACESpecBuilder.default_file_layout.training_input_filenames["config"]
        == "mace.yaml"
    )
    restored = MLFFSpec.model_validate_json(first.model_dump_json())
    assert restored.file_layout == first.file_layout
    assert restored.mlff_spec_id == second.mlff_spec_id


@pytest.mark.parametrize(
    "field",
    [
        "pretrained_model_filename", "evaluation_model_filename",
        "training_input_filenames", "training_output_filenames",
    ],
)
@pytest.mark.parametrize(
    "unsafe",
    ["", "../escape", "/absolute", "C:/absolute", "a\\b", "a//b", "a/./b"],
)
def test_layout_rejects_unsafe_names(field: str, unsafe: str) -> None:
    """Validate names before the writer adds any directory prefix.

    Parameters
    ----------
    field : str
        Filename field or filename mapping under test.
    unsafe : str
        Nonportable or nonrelative name that validation must reject.
    """
    data = MACESpecBuilder.default_file_layout.model_dump()
    data[field] = {"config": unsafe} if field.endswith("filenames") else unsafe
    with pytest.raises(ValidationError):
        MLFFFileLayout.model_validate(data)


def test_spec_requires_saved_layout(tmp_path: Path) -> None:
    """Reject a specification without layout instead of consulting defaults.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for a valid source checkpoint.
    """
    (tmp_path / "source.model").write_bytes(b"weights")
    spec = MACESpecBuilder(
        model_name="test",
        model_version="1",
        pretrained_model_dir=tmp_path,
        model_filename="source.model",
    ).build()
    data = spec.model_dump()
    del data["file_layout"]
    with pytest.raises(ValidationError, match="file_layout"):
        MLFFSpec.model_validate(data)
