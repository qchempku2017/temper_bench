"""Resolve native SevenNet model settings from the submitted checkpoint."""

from pathlib import Path

def prepare_config(config_path: Path, output_path: Path, *, bundle_root: Path) -> None:
    """Write training YAML with checkpoint architecture and recipe controls.

    Parameters
    ----------
    config_path : Path
        Submitted YAML containing training controls and the checkpoint path.
    output_path : Path
        Destination consumed by the native sevenn train command.
    bundle_root : Path
        Directory against which the YAML's relative checkpoint path is resolved.

    Returns
    -------
    None
        Writes the resolved YAML while preserving its bundle-relative paths.
    """
    import yaml
    from sevenn.util import load_checkpoint

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(str(bundle_root / config["train"]["continue"]["checkpoint"]))
    model = checkpoint.yaml_dict(mode="continue")["model"]
    model.update(config["model"])
    config["model"] = model
    output_path.write_text(yaml.safe_dump(config), encoding="utf-8")
