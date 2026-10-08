"""Tests for copied Calculator adapters and remote device resolution."""

from __future__ import annotations

import pytest
import sys
from types import ModuleType, SimpleNamespace

from temper.mlff.runtime import check_cuda as device
from temper.mlff.runtime.calculators import (
    deepmd,
    mace,
    mattersim,
    nep89,
    sevennet,
)


@pytest.fixture(autouse=True)
def cuda_runtime(monkeypatch):
    runtime = ModuleType("check_cuda")
    runtime.torch_device = lambda: "cuda"
    monkeypatch.setitem(sys.modules, "check_cuda", runtime)


def _config(**parameters):
    return {
        "model": "models/model",
        "calculator": {"parameters": parameters},
    }


def test_deepmd_uses_cuda(monkeypatch) -> None:
    calls = []
    package = ModuleType("deepmd")
    calculator = ModuleType("deepmd.calculator")
    calculator.DP = lambda **kwargs: calls.append(kwargs) or "dp"
    monkeypatch.setitem(sys.modules, "deepmd", package)
    monkeypatch.setitem(sys.modules, "deepmd.calculator", calculator)
    assert deepmd.build_calculator(_config(head="Energy")) == "dp"
    assert calls == [{"model": "models/model", "head": "Energy"}]


def test_mattersim_uses_cuda(monkeypatch) -> None:
    calls = []
    package = ModuleType("mattersim")
    forcefield = ModuleType("mattersim.forcefield")
    forcefield.MatterSimCalculator = (
        lambda **kwargs: calls.append(kwargs) or "mattersim"
    )
    monkeypatch.setitem(sys.modules, "mattersim", package)
    monkeypatch.setitem(sys.modules, "mattersim.forcefield", forcefield)
    assert mattersim.build_calculator(_config(custom=True)) == "mattersim"
    assert calls == [{"load_path": "models/model", "custom": True, "device": "cuda"}]


def test_sevennet_uses_cuda(monkeypatch) -> None:
    calls = []
    package = ModuleType("sevenn")
    calculator = ModuleType("sevenn.calculator")
    calculator.SevenNetCalculator = (
        lambda **kwargs: calls.append(kwargs) or "sevennet"
    )
    monkeypatch.setitem(sys.modules, "sevenn", package)
    monkeypatch.setitem(sys.modules, "sevenn.calculator", calculator)
    assert sevennet.build_calculator(_config(modal="mpa")) == "sevennet"
    assert calls == [{"model": "models/model", "modal": "mpa", "device": "cuda"}]


def test_mace_resolves_device_on_runner(monkeypatch) -> None:
    calls = []
    runtime_device = ModuleType("check_cuda")
    runtime_device.torch_device = lambda: "cuda"
    package = ModuleType("mace")
    calculators = ModuleType("mace.calculators")
    calculators.MACECalculator = (
        lambda **kwargs: calls.append(kwargs) or "mace"
    )
    monkeypatch.setitem(sys.modules, "check_cuda", runtime_device)
    monkeypatch.setitem(sys.modules, "mace", package)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)
    assert mace.build_calculator(_config(default_dtype="float64")) == "mace"
    assert calls == [
        {
            "model_paths": ["models/model"],
            "default_dtype": "float64",
            "device": "cuda",
        }
    ]


def test_nep_prefers_gpu_only_with_cuda_and_gpumd(monkeypatch) -> None:
    calls = []
    runtime_device = ModuleType("check_cuda")
    runtime_device.require_cuda = lambda: None
    calorine = ModuleType("calorine")
    calculators = ModuleType("calorine.calculators")
    calculators.GPUNEP = lambda *args, **kwargs: calls.append(
        ("gpu", args, kwargs)
    ) or "gpu"
    calculators.CPUNEP = lambda *args, **kwargs: calls.append(
        ("cpu", args, kwargs)
    ) or "cpu"
    monkeypatch.setitem(sys.modules, "check_cuda", runtime_device)
    monkeypatch.setitem(sys.modules, "calorine", calorine)
    monkeypatch.setitem(sys.modules, "calorine.calculators", calculators)
    monkeypatch.setattr(nep89.shutil, "which", lambda name: "/bin/gpumd")
    assert nep89.build_calculator(_config(timestep=1)) == "gpu"
    assert calls[0] == (
        "gpu",
        ("models/model",),
        {"timestep": 1},
    )

    calls.clear()
    monkeypatch.setattr(nep89.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="gpumd"):
        nep89.build_calculator(_config())
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
