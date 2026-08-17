# Data format and grouping

[Back to the project overview](../README.md) · [Data splitting](data-splitting.md) · [Roadmap](roadmap.md)

## Domain layout

A **domain** is one directory below the data root. The default data root is `./data`, so a domain named `sse_llzo` is normally laid out as follows:

```text
data/
└── sse_llzo/
    ├── metadata.json
    ├── sample_a.extxyz
    └── sample_b.extxyz
```

Every domain needs a `metadata.json` file with both top-level keys:

```json
{
  "info": ["... one entry per .extxyz file ..."],
  "groupings": ["... one or more grouping specifications ..."]
}
```

The loader reads top-level `.extxyz` files in the domain directory. It requires the number of files to equal the number of `info` entries, and uses the `filename` values in `info` to establish their order.

Each source frame must be readable by ASE and provide energy and forces. Stress is detected and recorded in the inventory, but is not required by the splitter.

## `info`: source-file inventory

Each entry represents one `extxyz` file and is loaded as an [`InfoEntry`](../src/temper/schemas/info.py:18). The required fields are:

- `name` — dataset name;
- `source` — where the dataset came from;
- `domain` — the domain directory name;
- `filename` — the `.extxyz` filename; and
- `system_type` — a non-empty list of type labels.

A typical metadata entry supplies the human-authored fields below. The loader needs `filename` in each metadata entry to establish file ordering. It then constructs the entry from the source file: [`InfoEntry.from_extxyz`](../src/temper/schemas/info.py:125) initially derives `name` from the filename stem, `domain` from the directory name, and `filename` from the file path. Metadata values passed to that helper can override those initially derived values, so keep them consistent with the source tree.

```json
{
  "name": "sample_a",
  "source": "publication or repository URL",
  "domain": "sse_llzo",
  "filename": "sample_a.extxyz",
  "system_type": ["ionic", "bulk"],
  "description": "Short description of the structures.",
  "first_principle_software": "VASP",
  "first_principles_settings": "Relevant calculation settings.",
  "theory_level": "PBE",
  "structure_generation_method": ["AIMD"],
  "additional_info": "Other provenance details."
}
```

The latter six fields are optional; missing optional fields produce warnings. Autodetection also inventories the source file's systems, frames per system, atom counts, formulas, stress availability, and other detected properties. The inventory follows the implementation's dpdata-like grouping: frames with the same atom count and Hill-formula composition belong to the same inferred system. Users normally should not manually supply the autodetected inventory fields.

## `groupings`: how files become split pools

Every item in `groupings` is passed to [`GroupedDomain.from_datadir_with_strategy`](../src/temper/schemas/group.py:97). It must contain `grouping_strategy` and may contain the strategy's keyword arguments, `add_extra_cross_tests`, and `specify_cross_tests`.

```json
{
  "grouping_strategy": "as_specified",
  "groups": {
    "low_temperature": ["sample_a.extxyz"],
    "high_temperature": ["sample_b.extxyz"]
  },
  "add_extra_cross_tests": false,
  "specify_cross_tests": {
    "low_temperature": ["high_temperature"]
  }
}
```

The registered strategy keys are:

- `by_every_file` — make one group per file, named with its filename stem;
- `as_specified` — use a `groups` mapping of group names to listed filenames;
- `all` — place all files in one group named `all`;
- `by_regex` — group filenames using `regex`, optionally formatting names with `group_name`; `strict` defaults to `true`;
- `by_property` — group filename conventions by `grouping_property` (`temperature`, `pressure`, `composition_1d_numerical`, `composition_string`, `u_specification`, or `mag_specification`); and
- `by_neb_generalization` — group NEB filenames into `endpoints_and_midpoint` and `intermediate` using their image indices.

See the concrete implementations and strategy arguments in [`src/temper/grouping/strategies.py`](../src/temper/grouping/strategies.py).

### Implemented cross-test behavior

When [`split_grouped_domain`](../src/temper/splitting/split.py:233) runs:

- if `specify_cross_tests` is not `null`, it takes precedence regardless of `add_extra_cross_tests`; the explicit group-to-extra-group mapping is used after removing each source group's self-reference and deduplicating its targets;
- otherwise, if `add_extra_cross_tests` is `true`, every group is assigned every other group as an extra test group; or
- otherwise, no extra cross tests are assigned.

The automatically generated other-group target ordering is not guaranteed because the implementation uses sets. Extra-test groups are recorded in each split result and can be included during export; their datasets are not merged into that result's own `test_set`.

## Next step

Use [`partition_domain_groups`](../src/temper/grouping/group.py:15) to load one domain's configured grouping strategies, then follow [Data splitting](data-splitting.md) to create and export splits.
