"""Resolve native SevenNet model settings from the submitted checkpoint."""

import argparse
from pathlib import Path

def prepare_config(config_path: Path, output_path: Path) -> None:
    """Write training YAML with checkpoint architecture and recipe controls.

    Parameters
    ----------
    config_path : Path
        Submitted YAML containing training controls and the checkpoint path.
    output_path : Path
        Destination consumed by the native sevenn train command.
    """
    import yaml
    from sevenn.util import load_checkpoint

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(config["train"]["continue"]["checkpoint"])
    model = checkpoint.yaml_dict(mode="continue")["model"]
    model.update(config["model"])
    config["model"] = model
    output_path.write_text(yaml.safe_dump(config), encoding="utf-8")


def main() -> None:
    """Resolve the submitted configuration before the native training command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_config(args.config, args.output)


if __name__ == "__main__":
    main()
