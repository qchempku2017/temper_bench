"""MatterSim fine-tuning defaults."""
from temper.mlff.spec_builders.base import BaseSpecBuilder
from temper.schemas.mlff_spec import MLFFImplementation


@BaseSpecBuilder.register(name="mattersim")
class MatterSimSpecBuilder(BaseSpecBuilder):
    """Build a MatterSim recipe from a local checkpoint directory.

    Batch size defaults to one while upstream issue #163 remains unresolved
    in the pinned 1.2.5 release; callers can override it for patched builds.
    """

    mlff_type = "mattersim"
    model_name = "MatterSim-v1-5M"
    model_version = "1.0.0"
    model_filename = "mattersim.pth"
    implementations = (MLFFImplementation(name="mattersim", version="1.2.5"),)
    epoch_key = "epochs"
    patience_key = "early_stop_patience"
    training_defaults = {
        "run_name": "temper",
        "epochs": 200,
        "batch_size": 1,
        "lr": 2e-4,
        "step_size": 10,
        "force_loss_ratio": 1.0,
        "stress_loss_ratio": 0.1,
        "seed": 42,
        "re_normalize": False,
        "scale_key": "per_species_forces_rms",
        "shift_key": "per_species_energy_mean_linear_reg",
        "init_scale": None,
        "init_shift": None,
        "trainable_scale": False,
        "trainable_shift": False,
        "ckpt_interval": 10,
        "cutoff": 5.0,
        "threebody_cutoff": 4.0,
    }
