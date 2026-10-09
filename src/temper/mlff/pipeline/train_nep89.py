"""Run one fresh TorchNEP fine-tuning job from a GPUMD potential."""

from __future__ import annotations


def run(
    config: str,
    train: str,
    validation: str,
    model: str,
    output_directory: str,
) -> None:
    """Fine-tune a GPUMD model with fixed benchmark controls.

    Parameters
    ----------
    config : str
        Path to the prepared nep.in training configuration.
    train, validation : str
        Paths to the labeled training and validation extxyz datasets.
    model : str
        Path to the pretrained NEP potential.
    output_directory : str
        Directory where TorchNEP writes checkpoints and nep_best.txt.

    Returns
    -------
    None
        Training outputs are written by TorchNEP; backend errors propagate.
    """
    from torchnep import train_nep

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
