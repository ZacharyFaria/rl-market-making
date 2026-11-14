"""Standalone module entrypoint for `python -m src.cli.execute`."""

import sys

from src.cli.main import main

if __name__ == "__main__":
    sys.exit(main(["execute", *sys.argv[1:]]))

