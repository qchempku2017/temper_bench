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
   `stress` strongly recommended but not compulsory. No other format allowed.
3. Each domain folder should contain a file `metadata.json` containing two sections as follows:
    ```json
    {
      "info": [...],
      "groupings": [...]
    } 
    ```
    The `info` section specifies information about each `extxyz` file, and the `groupings` section specifies
    how `extxyz` files should be grouped into unit datasets before train-test splitting on each dataset.

4. The `info` section should have the following structure:
    ```json
    [
    {
    "name": "dataset_name",
    "description": "dataset_description",
    "source": "data_set_source_or_author",
    "first_principle_software": "VASP/Abacus/quantum_expresso/CP2K/etc",
    "first_principles_settings": "Input settings for the first-principles software, such as INCAR for VASP",
    "theory_level": "PBE/PBESol/PBE+U/SCAN/etc",
    "system_type": [
         "covalent/ionic/metallic/molecular/etc, for bonding type, allow multiple",
         "2dmaterial/surface/bulk/cluster/transition/point_defect/etc, for structural type, allow multiple"
    ],
    "structure_generation_method":[
        "relaxtraj/pert/MLMD+DFT/AIMD/enumerate/etc"    
    ],
    "additional_info": "Describe other details required to reproduce the dataset, such as KPOINTS and POTCAR selection, etc."
    },
    ...]
    ```
    where each dictionary in the list represents the content of a single extxyz file following the
    dpdata convention, i.e.,

   - Each single extxyz file contains a dpdata.MultiSystems object.
   - Each system in the MultiSystems object represents a collection of structure frames with the same
   number of atoms for each element type, not merely the same chemical composition. For example,
   H200O100 and H150O75 are regarded as different systems.
   - Systems are concatenated along the frame dimension in the extxyz file, following the same order as
   they appear in num_frames_per_system, num_atoms_per_system, and formulas.
   See [dpdata documentation](https://docs.deepmodeling.com/projects/dpdata/en/stable/index.html) for more details.
   - A `name` for each extxyz file is recommended, though optional. If not provided, the name will be
   set to the corresponding extxyz file name stem.
   - The `source` and `system_type` field is required.
   - All other fields are optional, but recommended to be filled in as much as possible.

5. The `groupings` section should have the following structure:
```json
[
  {
    "grouping_strategy": "by_strategy1",
    "add_extra_cross_tests": true/false,
    "other_kwarg1": "value1",
    "other_kwarg2": "value2",
    ...
  },
  ...
]
```
For each grouping strategy's keyword arguments, please refer to the documentation of the corresponding
grouping strategy in the `grouping_strategies` module, then replace sections like `other_kwargs1` into
the corresponding keyword arguments required.