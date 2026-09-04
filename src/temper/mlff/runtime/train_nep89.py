#!/usr/bin/env python3
"""Run one fresh TorchNEP fine-tuning job from a GPUMD potential."""

from __future__ import annotations

import argparse
from importlib import import_module


def run(
    config: str,
    train: str,
    validation: str,
    model: str,
    output_directory: str,
) -> None:
    """Fine-tune with fixed benchmark runtime controls."""
    train_nep = getattr(import_module("torchnep"), "train_nep")

    train_nep(
        config,
        train,
        output_dir=output_directory,
        device="cuda",
        finetune_from=model,
        restart=False,
        recompute_q_scaler=False,
        slim_types=True,
        run_seed=42,
        valid_file=validation,
    )


def main() -> None:
    """Parse paths supplied by the generated submit script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-directory", required=True)
    arguments = parser.parse_args()
    run(
        arguments.config,
        arguments.train,
        arguments.validation,
        arguments.model,
        arguments.output_directory,
    )


if __name__ == "__main__":
    main()
