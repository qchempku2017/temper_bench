Systematic benchmark of fine-tuning capabilities of various machine-learned force-field architectures (MLFFs).

A complete benchmark is decomposed into three stages:
1. **Data splitting** - generation of training and testing datasets for the benchmark.  
   The same train-test splits will be reused for all MLFFs.
2. **Training and validation** - composition of Bohrium jobs, including:
   - Training MLFFs on the training datasets.
   - Validating MLFFs on the testing datasets.
3. **Metrics** - calculation of evaluation metrics for the testing results.

### Future plans
- A web platform for uploading custom datasets;
- Web UI for creating `info.json` and `groups.json` easily, including automatic info field filling based on uploaded files;
- Online train-test split repository;
- Results uploading;
- Benchmarking results visualization.

### Data format requirements
Uploaded data should follow a specific structure and comply with the following rules:
1. Each dataset should be stored in a separate folder representing a data domain, such as `sse_llzo/`,
  `surface_si/`, `amorphous_carbon/`, etc.
2. Labeled structure frames should be stored in `extxyz` files under the corresponding domain folder.
   Each `extxyz` file should contain exactly the keys `energy` and`forces`.
   `stress` strongly recommended but not compulsory.
3. **Optional**: Each domain folder should contain a file `metadata.json` with the following structure:
  ```json
  [
  {
    "name": "dataset_name",
    "description": "dataset_description",
    "source": "data_set_source_or_author",
    "datapath": "domain/data1.extxyz",
    "first_principle_software": "VASP/Abacus/quantum_expresso/CP2K/etc",
    "first_principles_settings": "Input settings for the first-principles software, such as INCAR for VASP",
    "theory_level": "PBE/PBESol/PBE+U/SCAN/etc",
    "system_type": [
         "covalent/ionic/metallic/molecular/etc",
         "2dmaterial/surface/bulk/cluster/transition/point_defect/etc"
    ],
    "structure_generation_method":[
        "relaxtraj/pert/MLMD+DFT/AIMD/enumerate/etc"    
    ],
    "has_energy": true,
    "has_forces": true,
    "has_stress": true,
    "has_other_properties": ["some_property"],
    "num_systems": 1,
    "num_frames_per_system": [100],
    "num_atoms_per_system": [50],
    "formulas": ["C50"],
    "additional_info": "Describe other details required to reproduce the dataset, such as KPOINTS and POTCAR selection, etc."
  },
   ...]
   ```

where each dictionary in the list represents the content of a single extxyz file following the
dpdata convention, i.e.,

- Each single extxyz file contains a dpdata.MultiSystems object.
- The name of extxyz file in datapath follows a specific format: `domain/data1.extxyz`, where `domain` is the name of the domain folder, and `data1` is the name of the extxyz file.
- Each system in the MultiSystems object represents a collection of structure frames with the same
number of atoms for each element type, not merely the same chemical composition. For example,
H200O100 and H150O75 are regarded as different systems.
- Systems are concatenated along the frame dimension in the extxyz file, following the same order as
they appear in num_frames_per_system, num_atoms_per_system, and formulas.
See [dpdata documentation](https://docs.deepmodeling.com/projects/dpdata/en/stable/index.html) for more details.

  **Note**: the content for `info.json` does not affect the data splitting process, and is therefore optional.
   However, it is still strongly recommended to provide this file following `dpdata` convention for better
   reproducibility and tracability.

4. **Required**: Each domain folder **must** contain a file `groups.json` to specify how extxyz files in the domain
should be grouped. The file should have the following structure:
```json
[
  {
    "grouping_strategy": "grouping_strategy1",
    "groups": {
          "group_name1": ["domain/data1.extxyz", "domain/data2.extxyz", ...]
          , ...
   },
    "add_extra_cross_tests": true/false
  },
  ...
]
```

where

- Each dictionary in the list represents a grouping type.
- `grouping_strategy` represents the criterion used to group the data, such as "by_temperature",
"by_U", "by_composition", or "by_size", etc.
- `groups` should be a `dict[str, list]`, where each inner list contains the names of the extxyz files
belonging to the same group.
- `extxyz` files belonging to the same group will be merged into a single data unit for performing
train/val/test splitting.
- `add_extra_cross_tests` is a boolean value that indicates whether additional cross-group tests should be performed.
If true, beyond testing within each group, data from other groups will also be used for testing,
and testing results will be reported separately for each group.
- Data from different domain folders will never be grouped nor cross-tested.