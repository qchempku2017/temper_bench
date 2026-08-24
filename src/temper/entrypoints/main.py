"""Main CLI of TEMPER."""
import argparse

from temper.entrypoints.split import add_split_parser, split_cli


def main_parser() -> argparse.ArgumentParser:
    """TEMPER bench main CLI options and arguments parser.

    Returns
    ------
    ArgumentParser
        TEMPER bench main CLI options and arguments parser.
    """
    parser = argparse.ArgumentParser(
        description="Temper bench: benchmarking tool for evaluating the performance of"
        "fine-tuning on machine-learned force fields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")

    _ = add_split_parser(subparsers)

    return parser


def main():
    """TEMPER bench main CLI."""
    parser = main_parser()
    args = parser.parse_args()
    if args.command == "split":
        sysexit = split_cli(args.config_file)
    else:
        parser.print_help()
        sysexit = 1
    raise SystemExit(sysexit)


if __name__ == "__main__":
    main()
