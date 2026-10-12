"""Tests for calculator construction and remote device resolution."""

from __future__ import annotations

import pytest
import sys
from types import ModuleType, SimpleNamespace

from temper.mlff.pipeline import cuda_utils as device
from temper.mlff.pipeline import evaluate_mlff


@pytest.fixture(autouse=True)
def cuda_runtime(monkeypatch):
    runtime = ModuleType("cuda_utils")
    runtime.torch_device = lambda: "cuda"
    runtime.require_cuda = lambda: None
    monkeypatch.setitem(sys.modules, "temper.mlff.pipeline.cuda_utils", runtime)


def _config(family, **parameters):
    return {
        "model": "pretrained_models/model",
        "calculator": {"identifier": family, "parameters": parameters},
    }


@pytest.mark.parametrize("family", ["dpa4", "dpa4c"])
def test_deepmd_uses_cuda(family, monkeypatch) -> None:
    calls = []
    package = ModuleType("deepmd")
    calculator = ModuleType("deepmd.calculator")
    calculator.DP = lambda **kwargs: calls.append(kwargs) or "dp"
    monkeypatch.setitem(sys.modules, "deepmd", package)
    monkeypatch.setitem(sys.modules, "deepmd.calculator", calculator)
    assert evaluate_mlff.build_calculator(_config(family, head="Energy")) == "dp"
    assert calls == [{"model": "pretrained_models/model", "head": "Energy"}]


def test_unknown_calculator_has_a_clear_error():
    with pytest.raises(ValueError, match="Unsupported MLFF type 'unknown'"):
        evaluate_mlff.build_calculator(_config("unknown"))


def test_mattersim_uses_cuda(monkeypatch) -> None:
    calls = []
    package = ModuleType("mattersim")
    forcefield = ModuleType("mattersim.forcefield")
    forcefield.MatterSimCalculator = (
        lambda **kwargs: calls.append(kwargs) or "mattersim"
    )
    monkeypatch.setitem(sys.modules, "mattersim", package)
    monkeypatch.setitem(sys.modules, "mattersim.forcefield", forcefield)
    assert evaluate_mlff.build_calculator(_config("mattersim", custom=True)) == "mattersim"
    assert calls == [{"load_path": "pretrained_models/model", "custom": True, "device": "cuda"}]


def test_sevennet_uses_cuda(monkeypatch) -> None:
    calls = []
    package = ModuleType("sevenn")
    calculator = ModuleType("sevenn.calculator")
    calculator.SevenNetCalculator = (
        lambda **kwargs: calls.append(kwargs) or "sevennet"
    )
    monkeypatch.setitem(sys.modules, "sevenn", package)
    monkeypatch.setitem(sys.modules, "sevenn.calculator", calculator)
    assert evaluate_mlff.build_calculator(_config("sevennet", modal="mpa")) == "sevennet"
    assert calls == [{"model": "pretrained_models/model", "modal": "mpa", "device": "cuda"}]


def test_mace_resolves_device_on_runner(monkeypatch) -> None:
    calls = []
    runtime_device = ModuleType("cuda_utils")
    runtime_device.torch_device = lambda: "cuda"
    runtime_device.require_cuda = lambda: None
    package = ModuleType("mace")
    calculators = ModuleType("mace.calculators")
    calculators.MACECalculator = (
        lambda **kwargs: calls.append(kwargs) or "mace"
    )
    monkeypatch.setitem(sys.modules, "temper.mlff.pipeline.cuda_utils", runtime_device)
    monkeypatch.setitem(sys.modules, "mace", package)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)
    assert evaluate_mlff.build_calculator(_config("mace", default_dtype="float64")) == "mace"
    assert calls == [
        {
            "model_paths": ["pretrained_models/model"],
            "default_dtype": "float64",
            "device": "cuda",
        }
    ]


def test_nep_prefers_gpu_only_with_cuda_and_gpumd(monkeypatch) -> None:
    calls = []
    runtime_device = ModuleType("cuda_utils")
    runtime_device.require_cuda = lambda: None
    runtime_device.torch_device = lambda: "cuda"
    calorine = ModuleType("calorine")
    calculators = ModuleType("calorine.calculators")
    calculators.GPUNEP = lambda *args, **kwargs: calls.append(
        ("gpu", args, kwargs)
    ) or "gpu"
    calculators.CPUNEP = lambda *args, **kwargs: calls.append(
        ("cpu", args, kwargs)
    ) or "cpu"
    monkeypatch.setitem(sys.modules, "temper.mlff.pipeline.cuda_utils", runtime_device)
    monkeypatch.setitem(sys.modules, "calorine", calorine)
    monkeypatch.setitem(sys.modules, "calorine.calculators", calculators)
    monkeypatch.setattr(evaluate_mlff.shutil, "which", lambda name: "/bin/gpumd")
    assert evaluate_mlff.build_calculator(_config("nep89", timestep=1)) == "gpu"
    assert calls[0] == (
        "gpu",
        ("pretrained_models/model",),
        {"timestep": 1},
    )

    calls.clear()
    monkeypatch.setattr(evaluate_mlff.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="gpumd"):
        evaluate_mlff.build_calculator(_config("nep89"))
    assert not calls


def test_torch_device_order_and_cuda_visibility(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: True)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert device.torch_device() == "cuda"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        device.torch_device()

@pytest.mark.parametrize("visible", ("", "-1"))
def test_hidden_cuda_never_falls_back(monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
    ))
    assert not device.cuda_available()
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        device.require_cuda()
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        device.torch_device()


def test_cpu_only_torch_raises(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
    ))
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        device.torch_device()
