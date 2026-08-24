"""Focused tests for filename grouping strategies and domain partitioning."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from temper.grouping.group import partition_domain_into_groups
from temper.grouping.strategies import (
    GROUPING_STRATEGIES,
    group_all,
    group_as_specified,
    group_by_every_file,
    group_by_neb_generalization,
    group_by_property,
    group_by_regex,
)
from temper.schemas.group import GroupedDomain
from temper.schemas.info import InfoEntry


def test_basic_grouping_strategies_and_their_validations() -> None:
    files = ["one.extxyz", "sub/two.extxyz"]
    assert group_all(files) == {"all": files}
    assert group_by_every_file(files) == {"one": ["one.extxyz"], "two": ["sub/two.extxyz"]}
    assert group_as_specified(files, {"chosen": ["one.extxyz"]}) == {"chosen": ["one.extxyz"]}

    with pytest.raises(ValueError, match="not found"):
        group_as_specified(files, {"chosen": ["absent.extxyz"]})
    with pytest.raises(ValueError, match="Duplicate file stem"):
        group_by_every_file(["a/sample.extxyz", "b/sample.extxyz"])


def test_regex_and_property_grouping_cover_strict_and_non_strict_modes() -> None:
    files = ["alpha_t_300_run.extxyz", "beta_T_600_run.extxyz", "gamma_t_600_run.extxyz", "unmatched.extxyz"]
    assert group_by_regex(files, r".*_t_(\d+)_.*", group_name="temperature_{0}", strict=False) == {
        "temperature_300": ["alpha_t_300_run.extxyz"],
        "temperature_600": ["gamma_t_600_run.extxyz"],
    }  # Sensitive to capitalization.
    with pytest.raises(ValueError, match="unmatched.extxyz"):
        group_by_regex(files, r".*_t_(\d+)_.*")
    assert group_by_property(files[:3], "temperature") == {
        "temperature_300": ["alpha_t_300_run.extxyz"],
        "temperature_600": ["beta_T_600_run.extxyz", "gamma_t_600_run.extxyz"],
    }  # Insensitive to capitalization.
    with pytest.raises(ValueError, match="not found"):
        group_by_property(files, "unknown")


def test_neb_generalization_groups_endpoints_midpoints_and_intermediates() -> None:
    files = [
        "reaction_a_fp_00.extxyz", "reaction_a_fp_01.extxyz", "reaction_a_fp_02.extxyz", "reaction_a_fp_03.extxyz", "reaction_a_fp_04.extxyz",
        "reaction_b_LOC_10.extxyz", "reaction_b_LOC_11.extxyz", "reaction_b_LOC_12.extxyz",
        "not_a_neb.extxyz",
    ]
    assert group_by_neb_generalization(files, strict=False) == {
        "endpoints_and_midpoint": ["reaction_a_fp_00.extxyz", "reaction_a_fp_02.extxyz", "reaction_a_fp_04.extxyz", "reaction_b_LOC_10.extxyz", "reaction_b_LOC_11.extxyz", "reaction_b_LOC_12.extxyz"],
        "intermediate": ["reaction_a_fp_01.extxyz", "reaction_a_fp_03.extxyz"],
    }
    with pytest.raises(ValueError, match="not_a_neb.extxyz"):
        group_by_neb_generalization(files)


def test_grouped_domain_factory_applies_registered_strategy(extxyz_domain: Path) -> None:
    with pytest.warns(UserWarning, match="Missing optional fields"):
        entries = [
            InfoEntry(name="alpha", source="unit-test", domain="demo_domain", filename="alpha_t_300_run.extxyz", system_type=["molecule"]),
            InfoEntry(name="beta", source="unit-test", domain="demo_domain", filename="beta_t_600_run.extxyz", system_type=["atom"]),
        ]
    domain = GroupedDomain.from_datadir_with_strategy(
        extxyz_domain, "by_property", grouping_property="temperature", info_entries=entries,
    )
    assert domain.groups == {
        "temperature_300": ["alpha_t_300_run.extxyz"],
        "temperature_600": ["beta_t_600_run.extxyz"],
    }
    with pytest.raises(ValueError, match="Unknown grouping strategy"):
        GroupedDomain.from_datadir_with_strategy(extxyz_domain, "unknown", info_entries=entries)


def test_partition_domain_into_groups_reuses_metadata_and_applies_each_configuration(
    extxyz_domain: Path, metadata_payload: dict,
) -> None:
    metadata_payload["groupings"] = [
        {"grouping_strategy": "all"},
        {"grouping_strategy": "by_property", "grouping_property": "temperature"},
    ]
    (extxyz_domain / "metadata.json").write_text(json.dumps(metadata_payload), encoding="utf-8")
    root = extxyz_domain.parent
    with pytest.warns(UserWarning, match="Missing optional fields"):
        grouped = partition_domain_into_groups("demo_domain", root_path=root)

    assert [domain.grouping_strategy for domain in grouped] == ["all", "by_property"]
    assert grouped[0].groups == {"all": ["alpha_t_300_run.extxyz", "beta_t_600_run.extxyz"]}
    assert grouped[1].groups == {
        "temperature_300": ["alpha_t_300_run.extxyz"],
        "temperature_600": ["beta_t_600_run.extxyz"],
    }
    assert grouped[0].info_entries == grouped[1].info_entries
    assert set(GROUPING_STRATEGIES) == {"by_every_file", "as_specified", "all", "by_regex", "by_property", "by_neb_generalization"}
