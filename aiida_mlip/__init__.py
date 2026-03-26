"""Machine learning interatomic potentials aiida plugin."""

from __future__ import annotations

from importlib.metadata import version

__version__ = version("aiida-mlip")

# Inclusive lower, exclusive upper
SUPPORTED_JANUS = ((0, 8, 3), (0, 9, 0))
