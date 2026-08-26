"""Compatibility import for the legacy-compatible NERF builder.

The implementation is maintained in PHEAT so rebuilding and scoring share one
structure contract.  This module remains as a stable import path for callers
that used the earlier QTF location.
"""

from pheat.nerf import NerfFolder

LegacyNerfFolder = NerfFolder

__all__ = ["LegacyNerfFolder", "NerfFolder"]
