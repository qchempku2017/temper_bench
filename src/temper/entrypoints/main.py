"""Main CLI of TEMPER."""
import argparse

from src.temper.entrypoints.split import add_split_parser, split_cli


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
        sysexit = split_cli(
            args.root_path,
            args.output_path,
            args.domains,
            args.split_repeats,
            args.seed,
            args.test_ratio,
            args.train_ratios,
            args.max_train_size,
            args.method,
            args.descriptor_k,
            args.descriptor_cutoff,
            args.descriptor_dtype,
            args.compute_descriptor_chunk_size,
            args.entropy_bandwidth,
            args.entropy_batch_size,
            args.device,
            args.gpu_device,
            args.numba_threads,
            args.write_validation,
            not args.no_write_extra_tests,
        )
    else:
        parser.print_help()
        sysexit = 1
    raise SystemExit(sysexit)


if __name__ == "__main__":
    main()
