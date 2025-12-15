"""
Core systolic array components.

This module contains the fundamental building blocks:
- PE: Processing Element (MAC unit)
- Tile: Combinational grid of PEs
- Mesh: Pipelined grid of Tiles
"""

from .pe import PE, PEWithShift

__all__ = ["PE", "PEWithShift"]
