"""Submit or inspect a tiny JobManager round trip from the local
host.

Usage: python validate_roundtrip.py submit|check WORKSPACE MODEL
RESOURCE_JSON. Inputs and downloaded outputs are owned by WORKSPACE. The
optional --family selects mace (default), dpa4, or dpa4c. DeepMD requires
the matching explicit JSON beside MODEL. Each submission uses one GPU.
check command writes concise evidence after successful hash verification
and remote cleanup; callers can then remove WORKSPACE after copying
roundtrip-result.json elsewhere.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from uuid import UUID

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from temper.job_management.manager import JobManager
from temper.job_management.schemas.resources import RemoteResource
from temper.mlff.spec_builders import mlff_spec_builder_factory
from temper.schemas.mlff_train_bundle import MLFFTrainBundle
from temper.schemas.train_unit import TrainingUnit

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("action", choices=("submit", "check"))
parser.add_argument("workspace")
parser.add_argument("model")
parser.add_argument("resource_file")
parser.add_argument(
    "--family", choices=("mace", "dpa4", "dpa4c"), default="mace"
)
args = parser.parse_args()
action, workspace, model, resource_file = (
    args.action,
    args.workspace,
    args.model,
    args.resource_file,
)
root = Path(workspace).resolve()
manager = JobManager(root / "jobs")
if action == "submit":
    data = root / "data" / "toy"
    data.mkdir(parents=True)
    frames = []
    for index in range(6):
        atoms = bulk("Cu", "fcc", a=3.60 + index * 0.015, cubic=True)
        atoms.positions[0, 0] += 0.01 * (index + 1)
        atoms.calc = EMT()
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=atoms.get_potential_energy(),
            forces=atoms.get_forces(),
            stress=atoms.get_stress(),
        )
        frames.append(atoms)
    for name, selected in (
        ("train", frames[:4]),
        ("val", frames[4:5]),
        ("test", frames[5:]),
    ):
        write(data / (name + ".extxyz"), selected)
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
        split_id=UUID("0c112fa7-ec27-5ee2-829d-ef1cb92e8238"),
        train_set="train.extxyz",
        val_set="val.extxyz",
        test_sets=("test.extxyz",),
        root_path=root / "data",
    ).with_dataset_hashes()
    checkpoint = Path(model).resolve()
    training = (
        {"max_num_epochs": 1, "batch_size": 1}
        if args.family == "mace"
        else {
            "numb_epoch": 1,
            "save_freq": 1,
            "disp_freq": 1,
            "training_data": {"batch_size": 1},
            "validation_data": {"batch_size": 1},
        }
    )
    spec = mlff_spec_builder_factory(args.family)(
        model_name="compatibility-smoke",
        model_version="20261027",
        pretrained_model_dir=checkpoint.parent,
        model_filename=checkpoint.name,
        training_parameters=training,
    ).build()
    record = manager.submit(
        MLFFTrainBundle(training_unit=unit, mlff_spec=spec),
        RemoteResource.from_file(resource_file),
    )
elif action == "check":
    records = manager.synchronize()
    if len(records) != 1:
        raise RuntimeError("Expected exactly one round-trip attempt.")
    record = records[0]
    if record.cleanup_status == "complete":
        assert record.computation_status == "success", record.model_dump_json()
        folder = manager.store.path(record)
        with np.load(folder / "testing_outputs/test_000.npz") as predictions:
            for key, shape in {
                "energies": (1,),
                "forces": (4, 3),
                "stresses": (1, 3, 3),
            }.items():
                assert predictions[key].shape == shape
                assert np.isfinite(predictions[key]).all()
        (root / "roundtrip-result.json").write_text(
            record.model_dump_json(indent=2), encoding="utf-8"
        )
else:
    raise ValueError("Action must be submit or check.")
print(
    json.dumps(
        {
            "submission": record.submission_status,
            "computation": record.computation_status,
            "cleanup": record.cleanup_status,
            "error": record.error,
            "attempt": record.attempt_folder_name,
            "dispatcher": record.dispatcher,
        },
        indent=2,
    )
)
