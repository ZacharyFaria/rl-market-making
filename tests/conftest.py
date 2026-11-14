"""Pytest configuration ensuring the local repository `src` package takes precedence."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path or sys.path[0] != ROOT:
    if ROOT in sys.path:
        sys.path.remove(ROOT)
    sys.path.insert(0, ROOT)

for mod_name in list(sys.modules):
    if mod_name == "src" or mod_name.startswith("src."):
        del sys.modules[mod_name]

