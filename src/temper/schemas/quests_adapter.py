"""Persisted configuration schema for QUESTS descriptor and entropy computation."""
from __future__ import annotations

import math
import multiprocessing as mp
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from src.temper.schemas.base import MSONableModel


class QuestsAdapterConfig(MSONableModel):
    """Typed configuration for the QUESTS adapter.

    Persists every descriptor, entropy, device, and reproducibility parameter
    consumed by the QUESTS adapter in ``src.temper.splitting.quests``. The
    QUESTS selection itself is deterministic (greedy maximum-information-gain
    selection conditioned on the descriptors selected so far), so no random
    seed is required or stored; see the adapter module documentation.

    Attributes:
        descriptor_k (int): Number of nearest neighbors ``k`` used by the
            QUESTS per-atom descriptor. The concatenated descriptor has
            ``2*k - 1`` columns. Must be at least 2.
        descriptor_cutoff (float): Cutoff radius (in Ångström) of the
            descriptor weight function. Must be positive.
        descriptor_dtype (Literal["float32", "float64"]): Floating-point dtype
            of the computed descriptors and, for the GPU route, of the torch
            tensors passed to the QUESTS backend.
        compute_descriptor_chunk_size (int): Number of frames processed at a
            time when computing descriptors. Must be positive. This
            allows chunked computation and prevents memory overflow. Defaults
            to 200.
        entropy_bandwidth (float): Bandwidth ``h`` of the Gaussian kernel used
            by the QUESTS entropy adapter. Must be positive.
        entropy_batch_size (int): Maximum batch size used by the QUESTS
            backend when batching distance computations. Must be positive.
        device (Literal["cpu", "gpu", "auto"]): Which QUESTS backend route to
            use. ``"cpu"`` never imports or initializes CUDA/torch;
            ``"gpu"`` requires an available CUDA device and raises
            :class:`src.temper.splitting.quests.QuestsUnavailableError`
            otherwise; ``"auto"`` uses the GPU route when a CUDA device is
            available and falls back to the CPU route otherwise (documented
            fallback).
        gpu_device (str | None): Optional torch device string (e.g.
            ``"cuda:0"``) used by the GPU route. Must be ``None`` when
            ``device == "cpu"``.
        numba_threads (int | None): Optional number of threads for the numba
            parallel sections of the CPU descriptor/entropy kernels. ``None``
            will use half of the available CPU threads (maximum 8). The results are
            deterministic regardless of this value.
    """

    model_config = ConfigDict(frozen=True)

    descriptor_k: int = Field(default=32, ge=2)
    descriptor_cutoff: float = Field(default=5.0, gt=0)
    descriptor_dtype: Literal["float32", "float64"] = "float32"
    compute_descriptor_chunk_size: int = Field(default=200, gt=0)
    entropy_bandwidth: float = Field(default=0.015, gt=0)
    entropy_batch_size: int = Field(default=20000, gt=0)
    device: Literal["cpu", "gpu", "auto"] = "auto"
    gpu_device: str | None = None
    numba_threads: int = Field(
        default_factory=lambda: min(8, max(1, mp.cpu_count() // 2)),
        ge=1,
        le=8,
    )

    @field_validator("descriptor_cutoff", "entropy_bandwidth")
    @classmethod
    def validate_finite_float(cls, value: float) -> float:
        """Reject non-finite numeric configuration values."""
        if not math.isfinite(value):
            raise ValueError(f"value must be finite, got {value}.")
        return value

    @field_validator("numba_threads", mode="before")
    @classmethod
    def clip_numba_threads(cls, value: int | None) -> int:
        """Default and cap the number of numba worker threads."""
        if value is None:
            return min(8, max(1, mp.cpu_count() // 2))
        if value < 1:
            raise ValueError(f"numba_threads must be positive when set, got {value}.")
        return min(8, value)

    @field_validator("gpu_device")
    @classmethod
    def validate_gpu_device(cls, value: str | None) -> str | None:
        """Require an explicit GPU device string to contain text."""
        if value is not None and not value.strip():
            raise ValueError("gpu_device must be a non-empty device string when set.")
        return value

    @model_validator(mode="after")
    def validate_device_consistency(self) -> "QuestsAdapterConfig":
        """Disallow GPU configuration when the CPU route is selected."""
        if self.device == "cpu" and self.gpu_device is not None:
            raise ValueError(
                "gpu_device must be None when device == 'cpu'; "
                "the CPU route never initializes CUDA."
            )
        return self
