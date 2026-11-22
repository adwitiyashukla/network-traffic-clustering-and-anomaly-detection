import argparse
import sys

from netmine.config import ConfigError, load_config
from netmine.data import DataError, ensure_dataset
from netmine.pipeline import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="netmine")
    parser.add_argument("command", choices=["download", "run"])
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        config = load_config(arguments.config)
    except ConfigError as error:
        print("config error: " + str(error), file=sys.stderr)
        return 2
    try:
        if arguments.command == "download":
            train_path, test_path = ensure_dataset(config)
            print(str(train_path))
            print(str(test_path))
            return 0
        run(config, verbose=not arguments.quiet)
    except DataError as error:
        print("data error: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
