"""Checkpoint selection without installing or executing model backends."""

import json
from importlib.resources import files

import pytest

from temper.mlff.spec_builders.base import pretrained_model_variants, pretrained_model_defaults
from temper.mlff.spec_builders import MACESpecBuilder, mlff_spec_builder_factory
from temper.schemas.utils import validate_submit_relative_path


CATALOG = json.loads(files("temper.mlff").joinpath(
    "default_pretrained_models.json"
).read_text(encoding="utf-8"))
ENTRIES = [
    (family, variant, entry)
    for family, variants in CATALOG["models"].items()
    for variant, entry in variants.items()
]


@pytest.mark.parametrize("family,variant,entry", ENTRIES)
def test_catalog_variants_support_finetuning(family, variant, entry, tmp_path):
    validate_submit_relative_path(entry["model_filename"])
    assert entry["source_url"].startswith("https://")
    assert entry["download_url"].startswith("https://")
    assert type(entry["estimated_num_parameters"]) is int
    assert entry["estimated_num_parameters"] > 0
    assert entry["parameter_count_notes"]
    assert "parameter_count" not in entry
    assert "unavailable" not in entry
    assert "finetuning_unavailable" not in entry
    builder_type = mlff_spec_builder_factory(family)
    assert variant in builder_type.available_variants()
    builder = builder_type(variant=variant, pretrained_model_dir=tmp_path)
    checkpoint = tmp_path / entry["model_filename"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"test checkpoint")
    assert builder.model_name == entry["model_name"]
    assert builder.model_version == entry["model_version"]
    spec = builder.build()
    assert spec.pretrained_model.name == entry["model_name"]
    assert spec.pretrained_model.artifacts["model"].path == checkpoint
    assert spec.testing_parameters == entry.get("testing_parameters", {})
    builder.training_parameters = {}
    assert builder.build().training_parameters is not None


@pytest.mark.parametrize("kwargs", [
    {}, {"model_name": "custom"},
    {"variant": ""}, {"variant": "unknown"},
    {"variant": "mpa-0-medium", "model_filename": "other.model"},
    {"variant": "mpa-0-medium", "model_name": "other"},
    {"variant": "mpa-0-medium", "model_version": "other"},
    {"model_name": "custom", "model_version": " ", "model_filename": "a"},
    {"model_name": "custom", "model_version": "1", "model_filename": "../a"},
])
def test_ambiguous_or_incomplete_selection_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MACESpecBuilder(**kwargs)


def test_unknown_variant_lists_choices():
    with pytest.raises(ValueError, match="mpa-0-medium"):
        MACESpecBuilder(variant="typo")
    with pytest.raises(ValueError, match="No pretrained model catalog"):
        pretrained_model_variants("missing-family")


def test_custom_identity_is_independent_of_other_instances(tmp_path):
    (tmp_path / "model").write_bytes(b"weights")
    first = MACESpecBuilder(model_name="small", model_version="1",
                            model_filename="model", pretrained_model_dir=tmp_path)
    second = MACESpecBuilder(model_name="large", model_version="2",
                             model_filename="model", pretrained_model_dir=tmp_path)
    assert first.build().pretrained_model.name == "small"
    assert second.build().pretrained_model.version == "2"
    assert first.build().mlff_spec_id != second.build().mlff_spec_id
    assert "model_name" not in MACESpecBuilder.__dict__


def test_variant_selection_does_not_require_checkpoint_until_build(tmp_path):
    builder = MACESpecBuilder(variant="mpa-0-medium", pretrained_model_dir=tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        builder.build()


def test_catalog_and_calculator_options_are_independent(monkeypatch):
    # Exercise required calculator options independently of release availability.
    entry = pretrained_model_defaults("mace", "mpa-0-medium")
    entry["testing_parameters"] = {"head": "omat_pbe"}
    from copy import deepcopy
    monkeypatch.setattr(
        "temper.mlff.spec_builders.base.pretrained_model_defaults",
        lambda *_: deepcopy(entry),
    )
    from temper.mlff.spec_builders import base
    first = base.pretrained_model_defaults("mace", "test-head")
    first["testing_parameters"]["head"] = "changed"
    options = {"default_dtype": "float64"}
    builder = MACESpecBuilder(variant="test-head", testing_parameters=options)
    options["default_dtype"] = "float32"
    assert builder.testing_parameters == {"head": "omat_pbe", "default_dtype": "float64"}
    with pytest.raises(ValueError, match="requires testing_parameters"):
        MACESpecBuilder(variant="test-head", testing_parameters={"head": "omol"})


def test_aissquare_omat_and_matpes_releases_are_available():
    for family in ("dpa4", "dpa4c"):
        entry = pretrained_model_defaults(family, "nano-omat24-v20260911")
        assert "aissquare.com" in entry["source_url"]
        assert entry["download_url"].endswith(f"{family.upper()}-Nano-OMat24-v20260911.pt")
    assert pretrained_model_defaults("dpa4", "max-omat24-v20260911")["estimated_num_parameters"] == 99010408
    assert pretrained_model_defaults("dpa4", "mini-matpes-v20260628")["pretraining_data"] == "MatPES R2SCAN 2025.2"


@pytest.mark.parametrize("family,variant", [
    ("dpa4", "nano-omol25-100m-v20260820"),
    ("mace", "mh-1-omat_pbe"),
    ("sevennet", "omni-mpa"),
])
def test_unsupported_training_presets_are_not_catalog_choices(family, variant):
    with pytest.raises(ValueError, match="Unknown"):
        pretrained_model_defaults(family, variant)


def test_variant_and_explicit_bundle_have_same_identity(tmp_path):
    entry = pretrained_model_defaults("mace", "mpa-0-medium")
    checkpoint = tmp_path / entry["model_filename"]
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    preset = MACESpecBuilder(variant="mpa-0-medium", pretrained_model_dir=tmp_path).build()
    explicit = MACESpecBuilder(pretrained_model_dir=tmp_path, **{
        field: entry[field] for field in ("model_name", "model_version", "model_filename")
    }).build()
    assert preset.mlff_spec_id == explicit.mlff_spec_id
