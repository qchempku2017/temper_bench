"""SevenNet checkpoint fine-tuning defaults."""
from temper.mlff.spec_builders.base import BaseSpecBuilder
from temper.schemas.mlff_spec import MLFFImplementation


@BaseSpecBuilder.register(name="sevennet")
class SevenNetSpecBuilder(BaseSpecBuilder):
    """Build a SevenNet-0 recipe; the runtime resumes its pretrained weights."""

    mlff_type = "sevennet"
    model_name = "SevenNet-0"
    model_version = "11July2024"
    model_filename = "sevennet.pth"
    implementations = (MLFFImplementation(name="sevenn", version="0.13.0"),)
    epoch_key = "epoch"
    training_defaults = {
        "random_seed": 1,
        "epoch": 100,
        "loss": "Huber",
        "loss_param": {"delta": 0.01},
        "optimizer": "adam",
        "optim_param": {"lr": 0.004},
        "scheduler": "exponentiallr",
        "scheduler_param": {"gamma": 0.99},
        "force_loss_weight": 1.0,
        "stress_loss_weight": 0.01,
        "per_epoch": 10,
        "batch_size": 4,
        "data_divide_ratio": 0.1,
        "train_shift_scale": False,
        "train_denominator": False,
    }
