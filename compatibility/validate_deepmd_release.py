"""Test a checkpoint/JSON pair with native DeepMD, independently of
TEMPER."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import distributions, version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Validate native DeepMD training, export, and ASE inference.

    Command-line arguments select dpa4 or dpa4c, the checkpoint, the
    matching JSON, and a new output directory. All paths are resolved
    from the current directory. Remote execution requires Slurm or PBS;
    --local-wsl permits an explicitly requested local WSL2 test. Exactly
    one GPU must be visible.

    --molecular uses four slightly displaced neutral singlet H2 molecules
    in a 20 angstrom periodic cell and supplies charge_spin=[0, 1] to
    training and ASE inference. The large cell separates periodic images.
    Otherwise the test uses periodic Cu cells. EMT labels exercise data
    plumbing only; neither toy dataset measures scientific accuracy.

    Use --resume-export after an export failure to reuse completed training
    with the same checkpoint, configuration and --molecular setting.
    Previous result metadata is
    retained, logs are appended, and the calculation is not trained again.

    Results and package versions are saved in result.json, including
    failed stages. Successful runs remove generated data, caches and
    model exports; failures retain their working directory for
    diagnosis. Inputs are never modified. Energies use eV, forces
    eV/angstrom, and ASE stress eV/angstrom^3.

    Raises
    ------
    ValueError
        If execution is outside an allowed GPU environment, the output
        exists, or the supplied configuration uses a preset or lacks an
        element map.
    subprocess.CalledProcessError
        If native fine-tuning or freeze fails; logs remain in the output
        folder.
    subprocess.TimeoutExpired
        If training or freeze exceeds the requested per-command timeout.
    RuntimeError
        If GPU visibility, exported output, or finite prediction checks
        fail.
    OSError
        If input, output or executable files cannot be accessed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("dpa4", "dpa4c"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--local-wsl", action="store_true")
    parser.add_argument("--resume-export", action="store_true")
    parser.add_argument("--molecular", action="store_true")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID")
    local_wsl = args.local_wsl and "microsoft" in platform.release().lower()
    if not job_id and not local_wsl:
        raise ValueError(
            "Use a scheduler GPU allocation or explicit --local-wsl."
        )
    checkpoint = args.checkpoint.resolve()
    configuration = args.configuration.resolve()
    output = args.output.resolve()
    previous = None
    if args.resume_export:
        previous = json.loads((output / "result.json").read_text())
        if (
            previous.get("train") != "passed"
            or previous["checkpoint_sha256"] != _sha256(checkpoint)
            or previous["configuration_sha256"] != _sha256(configuration)
            or previous.get("molecular", False) != args.molecular
        ):
            raise ValueError(
                "Resume requires completed training with the same inputs"
                " and molecular mode."
            )
    elif output.exists():
        raise ValueError(
            "Choose a new output directory to preserve prior evidence."
        )
    source = json.loads(configuration.read_text(encoding="utf-8"))
    if "preset" in source["model"] or not source["model"].get("type_map"):
        raise ValueError("Use an explicit model JSON with its full type_map.")
    output.mkdir(parents=True, exist_ok=args.resume_export)
    work = output / "work"
    work.mkdir(exist_ok=args.resume_export)
    cache = Path(tempfile.mkdtemp(prefix="temper-deepmd-"))
    environment = dict(os.environ)
    for name, child in {
        "TRITON_CACHE_DIR": "triton",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TORCH_EXTENSIONS_DIR": "extensions",
        "CUDA_CACHE_PATH": "cuda",
        "XDG_CACHE_HOME": "xdg",
    }.items():
        environment.setdefault(name, str(cache / child))
        os.environ.setdefault(name, environment[name])
    environment["DEVICE"] = "cuda"
    os.environ["DEVICE"] = "cuda"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment.setdefault(name, "4")
    result: dict[str, object] = {
        "family": args.family,
        "job_id": job_id,
        "local_wsl": local_wsl,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
        "configuration": configuration.name,
        "configuration_sha256": _sha256(configuration),
        "deepmd": version("deepmd-kit"),
        "python": sys.version,
        "prefix": sys.prefix,
        "stage": "initialization",
        "status": "failed",
        "packages": {d.metadata["Name"]: d.version for d in distributions()},
        "retained_cache": str(cache),
        "molecular": args.molecular,
        "validator_sha256": _sha256(Path(__file__)),
    }
    start = time.monotonic()
    if previous is not None:
        result["train"] = "passed"
        result["previous_attempt"] = previous
    try:
        import numpy as np
        import torch
        from ase import Atoms
        from ase.build import bulk
        from ase.calculators.emt import EMT
        from deepmd.calculator import DP

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one CUDA GPU must be visible.")
        result["gpu"] = torch.cuda.get_device_name()
        result["torch"] = torch.__version__
        result["torch_cuda"] = torch.version.cuda
        frames = []
        for index in range(4):
            if args.molecular:
                atoms = Atoms(
                    "H2", positions=[[10, 10, 10], [10.75, 10, 10]],
                    cell=[20, 20, 20], pbc=True,
                )
                atoms.info["charge_spin"] = np.array([0, 1])
            else:
                atoms = bulk(
                    "Cu", "fcc", a=3.60 + index * 0.015, cubic=True
                )
            atoms.positions[0, 0] += 0.01 * (index + 1)
            atoms.calc = EMT()
            frames.append(atoms)
        for name, selected in (("train", frames[:2]), ("valid", frames[2:3])):
            directory = work / name
            arrays = directory / "set.000"
            arrays.mkdir(parents=True, exist_ok=args.resume_export)
            element = "H" if args.molecular else "Cu"
            (directory / "type_map.raw").write_text(element + "\n")
            (directory / "type.raw").write_text("0\n" * len(selected[0]))
            if args.molecular:
                np.save(
                    arrays / "charge_spin.npy",
                    np.asarray([a.info["charge_spin"] for a in selected]),
                )
            for label, values in {
                "coord": [a.positions.reshape(-1) for a in selected],
                "box": [a.cell.array.reshape(-1) for a in selected],
                "energy": [a.get_potential_energy() for a in selected],
                "force": [a.get_forces().reshape(-1) for a in selected],
                "virial": [
                    (-a.get_stress(voigt=False) * a.get_volume()).reshape(-1)
                    for a in selected
                ],
            }.items():
                np.save(arrays / (label + ".npy"), np.asarray(values))
        config = deepcopy(source)
        config["training"] = {
            "numb_epoch": 1,
            "save_freq": 1,
            "disp_freq": 1,
            "seed": 42,
            "training_data": {"systems": ["train"], "batch_size": 1},
            "validation_data": {
                "systems": ["valid"],
                "batch_size": 1,
                "numb_btch": 1,
            },
            "save_ckpt": "model.ckpt",
            "save_dir": "checkpoints",
        }
        config["learning_rate"] = {
            "type": "exp",
            "start_lr": 1e-4,
            "stop_lr": 1e-6,
        }
        if not args.resume_export:
            (work / "input.json").write_text(json.dumps(config, indent=2))
        executable = str(Path(sys.executable).with_name("dp"))
        backend = "--pt" if args.family == "dpa4" else "--pt-expt"
        commands = [
            (
                "train",
                [
                    executable,
                    backend,
                    "train",
                    "input.json",
                    "--finetune",
                    str(checkpoint),
                    "--use-pretrain-script",
                    "-o",
                    "adapted.json",
                ],
            ),
            (
                "freeze",
                [
                    executable,
                    backend,
                    "freeze",
                    "-c",
                    "model.ckpt.pt",
                    "-o",
                    "evaluation",
                ]
                + (
                    ["--lower-kind", "graph"] if args.family == "dpa4c" else []
                ),
            ),
        ]
        for stage, command in commands:
            if stage == "train" and args.resume_export:
                continue
            result["stage"] = stage
            result[stage + "_command"] = command
            with (output / (stage + ".log")).open("a") as log:
                subprocess.run(
                    command,
                    cwd=work,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=True,
                )
            result[stage] = "passed"
        adapted = json.loads((work / "adapted.json").read_text())
        if adapted["model"]["type_map"] != source["model"]["type_map"]:
            raise RuntimeError(
                "Fine-tuning changed the requested element order."
            )
        result["stage"] = "ASE inference"
        export = work / "evaluation.pt2"
        frames[3].calc = DP(model=str(export))
        energy = frames[3].get_potential_energy()
        forces = frames[3].get_forces()
        stress = frames[3].get_stress(voigt=False)
        if not all(np.isfinite(v).all() for v in (energy, forces, stress)):
            raise RuntimeError("Nonfinite ASE predictions.")
        if forces.shape != (len(frames[3]), 3) or stress.shape != (3, 3):
            raise RuntimeError("Unexpected force or stress shape.")
        result["energy_eV"] = float(energy)
        result["force_shape"] = list(forces.shape)
        result["stress_shape"] = list(stress.shape)
        export.unlink()
        result["export_cleanup"] = (
            "passed" if not export.exists() else "failed"
        )
        shutil.copyfile(work / "input.json", output / "input.json")
        shutil.copyfile(work / "adapted.json", output / "adapted.json")
        shutil.rmtree(work)
        shutil.rmtree(cache)
        result.pop("retained_cache")
        result["status"] = "passed"
    except Exception as error:
        result["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        (output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
