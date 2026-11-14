"""Standalone module entrypoint for `python -m src.cli.evaluate`."""

import sys

from src.cli.main import main

if __name__ == "__main__":
    sys.exit(main(["evaluate", *sys.argv[1:]]))

