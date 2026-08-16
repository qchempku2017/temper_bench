# Data format and grouping

[Back to the project overview](../README.md) · [Data splitting](data-splitting.md) · [Roadmap](roadmap.md)

## Dataset layout

Uploaded data must follow these rules:

1. Store each dataset in a separate directory representing a data domain, such as `sse_llzo/`, `surface_si/`, or `amorphous_carbon/`.
2. Store labeled structure frames in `extxyz` files under the corresponding domain directory. Each file must contain the keys `energy` and `forces`. `stress` is strongly recommended but optional. No other structure-data format is supported.
3. Add a `metadata.json` file to each domain directory with two sections:

   ```json
   {
     "info": [...],
     "groupings": [...]
   }
   ```

   `info` describes each `extxyz` file. `groupings` defines how those files are grouped into unit datasets before train-test splitting.

## The `info` section

The `info` section has this structure:

```json
[
  {
    "name": "dataset_name",
    "description": "dataset_description",
    "source": "dataset_source_or_author",
    "filename": "dataset_filename.extxyz",
    "first_principle_software": "VASP/Abacus/quantum_expresso/CP2K/etc",
    "first_principles_settings": "Input settings for the first-principles software, such as INCAR for VASP",
    "theory_level": "PBE/PBESol/PBE+U/SCAN/etc",
    "system_type": [
      "covalent/ionic/metallic/molecular/etc, for bonding type, allow multiple",
      "2dmaterial/surface/bulk/cluster/transition/point_defect/etc, for structural type, allow multiple"
    ],
    "structure_generation_method": [
      "relaxtraj/pert/MLMD+DFT/AIMD/enumerate/etc"
    ],
    "additional_info": "Other details required to reproduce the dataset, such as KPOINTS and POTCAR selection"
  }
]
```

Each object describes one `extxyz` file following the dpdata convention:

- A single `extxyz` file contains a `dpdata.MultiSystems` object.
- Each system in that object is a collection of structure frames with the same number of atoms for every element type, not merely the same chemical composition. For example, `H200O100` and `H150O75` are different systems.
- Systems are concatenated along the frame dimension in the same order in which they appear in `num_frames_per_system`, `num_atoms_per_system`, and `formulas`. See the [dpdata documentation](https://docs.deepmodeling.com/projects/dpdata/en/stable/index.html) for details.
- `name` is recommended but optional. When omitted, it is set to the corresponding `extxyz` filename stem.
- `source`, `system_type`, and `filename` are required. Although `filename` can be inferred from the file itself, it must be explicit so that the order of `info` entries is preserved while loading the corresponding files.
- Every other field is optional, but should be completed as fully as possible to document and reproduce the dataset.

## The `groupings` section

The `groupings` section has this structure:

```json
[
  {
    "grouping_strategy": "by_strategy1",
    "add_extra_cross_tests": true,
    "other_kwarg1": "value1",
    "other_kwarg2": "value2"
  }
]
```

Replace placeholder fields such as `other_kwarg1` with the keyword arguments required by the selected strategy. The available strategy schemas are defined in [`src/temper/schemas/group.py`](../src/temper/schemas/group.py), and grouping behavior is implemented in [`src/temper/utils/grouping.py`](../src/temper/grouping/strategies.py).

### Cross-test behavior

- With `add_extra_cross_tests: true`, the model fine-tuned on each data group is tested on every other data group.
- With `add_extra_cross_tests: false`, the fine-tuned model is tested only on the test set split from the same data group by default.
- When `add_extra_cross_tests` is `false` and `specify_cross_tests` is also set, cross-tests are run only on the specified data groups. See [`src/temper/schemas/group.py`](../src/temper/schemas/group.py) for the schema details.

## Next step

After domain files and grouping metadata are prepared, see [Data splitting](data-splitting.md) for the split schemas, API, semantics, persistence, and export workflow.
