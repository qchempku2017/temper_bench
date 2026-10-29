"""Validate one MLFF family inside a scheduler-provided single-GPU
allocation."""

import os
import argparse
import platform

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "family",
    choices=("mace", "mattersim", "nep89", "sevennet", "dpa4", "dpa4c"),
)
parser.add_argument("model")
parser.add_argument("--local-wsl", action="store_true")
arguments = parser.parse_args()

job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID")
local_wsl = arguments.local_wsl and "microsoft" in platform.release().lower()
assert job_id or local_wsl, "Use a scheduler allocation or explicit local WSL"
from pathlib import Path
import sys, json, uuid, hashlib
from importlib.metadata import version, distributions, distribution
import numpy as np
import torch

assert torch.cuda.is_available()
assert (
    torch.cuda.device_count() == 1
), "Expose only the single GPU allocated by the scheduler"
print(
    "GPU", torch.cuda.get_device_name(), "TORCH", torch.__version__, flush=True
)
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from temper.schemas.train_unit import TrainingUnit
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.mlff.spec_builders import mlff_spec_builder_factory
from temper.mlff.pipeline.runner import run_pipeline, prepare_bundle
import temper

family, model = arguments.family, arguments.model
root = Path.cwd() / ("smoke-" + family)
root.mkdir()
data = root / "data" / "toy"
data.mkdir(parents=True)
frames = []
for i in range(6):
    atoms = bulk("Cu", "fcc", a=3.60 + i * 0.015, cubic=True)
    atoms.positions[0, 0] += 0.01 * (i + 1)
    atoms.calc = EMT()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    atoms.calc = SinglePointCalculator(
        atoms, energy=energy, forces=forces, stress=stress
    )
    frames.append(atoms)
write(data / "train.extxyz", frames[:4])
write(data / "val.extxyz", frames[4:5])
write(data / "test.extxyz", frames[5:])
unit = TrainingUnit(
    domain="toy",
    grouping_strategy="as_specified",
    group_name="copper",
    method="random",
    repeat_id=0,
    train_n_frames=4,
    val_n_frames=1,
    test_n_frames=1,
    train_n_atoms=16,
    val_n_atoms=4,
    test_n_atoms=4,
    split_id=uuid.UUID("0c112fa7-ec27-5ee2-829d-ef1cb92e8238"),
    train_set="train.extxyz",
    val_set="val.extxyz",
    test_sets=("test.extxyz",),
    root_path=root / "data",
)
overrides = {
    "mace": {"max_num_epochs": 1, "batch_size": 1},
    "mattersim": {"epochs": 1, "batch_size": 2},
    "sevennet": {"epoch": 1, "batch_size": 1, "per_epoch": 1},
    "nep89": {"epoch": 1, "batch": 1},
    "dpa4": {"numb_epoch": 1, "save_freq": 1, "disp_freq": 1},
    "dpa4c": {"numb_epoch": 1, "save_freq": 1, "disp_freq": 1},
}
model = Path(model).resolve()
spec = mlff_spec_builder_factory(family)(
    model_name="deployment-smoke",
    model_version="20261026",
    model_filename=model.name,
    pretrained_model_dir=model.parent,
    training_parameters=overrides[family],
).build()
unit = unit.with_dataset_hashes()
bundle = MLFFTrainBundle(training_unit=unit, mlff_spec=spec)
folder, packaged = bundle.write_submit_folder(root / "bundle")
if family in ("dpa4", "dpa4c"):
    adapter = prepare_bundle(folder / "bundle.json")
    config_path = folder / adapter.input_path("config")
    config = json.loads(config_path.read_text())
    config["model"].pop("type")
    config["training"]["optimizer"] = config.pop("optimizer")
    config_path.write_text(json.dumps(config))
run_pipeline(folder / "bundle.json")
pred = np.load(folder / "testing_outputs/test_000.npz")
for key in ("energies", "forces", "stresses"):
    assert np.isfinite(pred[key]).all(), key
assert pred["energies"].shape == (1,)
assert pred["forces"].shape == (4, 3)
assert pred["stresses"].shape == (1, 3, 3)
trained = folder / packaged.files.training_output_files["trained_model"]
assert trained.is_file() and trained.stat().st_size > 0
result = {
    "family": family,
    "job_id": job_id,
    "local_wsl": local_wsl,
    "gpu": torch.cuda.get_device_name(),
    "visible_gpu_count": torch.cuda.device_count(),
    "torch": torch.__version__,
    "training": "passed",
    "ASE_energy_forces_stress": "passed",
    "model_bytes": trained.stat().st_size,
    "environment_prefix": sys.prefix,
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "temper": version("temper-bench"),
    "training_parameters": overrides[family],
    "pretrained_sha256": (
        hashlib.file_digest(model.open("rb"), "sha256").hexdigest()
    ),
    "prediction_shapes": {
        key: list(pred[key].shape)
        for key in ("energies", "forces", "stresses")
    },
}
if family in ("dpa4", "dpa4c"):
    assert not (folder / packaged.files.evaluation_model).exists()
    result["freeze_and_cleanup"] = "passed"
    result["legacy_configuration_normalized"] = True
result["packages"] = {
    item.metadata["Name"]: version(item.metadata["Name"])
    for item in distributions()
}
source_root = Path(temper.__file__).parent
result["temper_source_files"] = {
    str(path.relative_to(source_root)): (
        hashlib.sha256(path.read_bytes()).hexdigest()
    )
    for path in sorted(source_root.rglob("*.py"))
}
result["temper_base_revision"] = os.environ.get("TEMPER_BASE_REVISION")
source_archive = Path(
    os.environ.get(
        "TEMPER_SOURCE_ARCHIVE", source_root.parent.parent / "source.tar.gz"
    )
)
result["temper_source_archive_sha256"] = (
    hashlib.sha256(source_archive.read_bytes()).hexdigest()
    if source_archive.is_file()
    else None
)
result["dependency_source_files"] = {}
for package, names in {
    "mace-torch": ["mace/cli/run_train.py"],
    "mattersim": [
        "mattersim/forcefield/potential.py",
        "mattersim/datasets/utils/converter.py",
    ],
    "calorine": ["calorine/calculators/gpunep.py"],
}.items():
    if package not in result["packages"]:
        continue
    installed = distribution(package)
    for name in names:
        path = Path(installed.locate_file(name))
        result["dependency_source_files"][name] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
(root / "result.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
