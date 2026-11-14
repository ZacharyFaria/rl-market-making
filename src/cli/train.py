"""Standalone module entrypoint for `python -m src.cli.train`."""

import sys

from src.cli.main import main

if __name__ == "__main__":
    sys.exit(main(["train", *sys.argv[1:]]))

