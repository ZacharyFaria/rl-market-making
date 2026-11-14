"""Standalone module entrypoint for `python -m src.cli.sanity_check`."""

import sys

from src.cli.main import main

if __name__ == "__main__":
    sys.exit(main(["sanity-check", *sys.argv[1:]]))

