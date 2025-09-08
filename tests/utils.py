"""Utility functions for tests."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path


@contextlib.contextmanager
def chdir(path):
    """Change working directory and return to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
