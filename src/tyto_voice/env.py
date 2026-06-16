"""Minimal .env loader, no dependencies.

Reads ``KEY=VALUE`` lines from a ``.env`` file (searching the current directory
and its parents) into ``os.environ``. Existing environment variables always win,
so exported values are never overwritten. Comments and blank lines are ignored.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(filename: str = ".env") -> Path | None:
    """Load the nearest ``.env`` into os.environ. Returns the path used, if any."""
    path = _find(filename)
    if path is None:
        return None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return path


def _find(filename: str) -> Path | None:
    for directory in [Path.cwd(), *Path.cwd().parents]:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None
