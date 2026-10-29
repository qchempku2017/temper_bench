"""Launch MatterSim 1.2.5 with final-step weights instead of averaged
weights.

The native training CLI does not expose ema_decay. This process-local
override sets decay to zero before its Potential is constructed; no
installed source is changed. All command-line arguments are passed to
the native CLI unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable
import runpy


def _main() -> None:
    import torch
    from torch_ema import ExponentialMovingAverage
    from mattersim.forcefield import potential

    class _FinalStepAverage(ExponentialMovingAverage):
        def __init__(
            self,
            parameters: Iterable[torch.Tensor],
            decay: float,
            use_num_updates: bool = True,
        ) -> None:
            # Decay zero makes each update retain the current weights
            # exactly.
            super().__init__(
                parameters, decay=0.0, use_num_updates=use_num_updates
            )

    potential.ExponentialMovingAverage = _FinalStepAverage
    try:
        runpy.run_module(
            "mattersim.training.finetune_mattersim", run_name="__main__"
        )
    finally:
        potential.ExponentialMovingAverage = ExponentialMovingAverage


if __name__ == "__main__":
    _main()
