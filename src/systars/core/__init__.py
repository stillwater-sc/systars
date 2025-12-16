"""
Core systolic matmul array components.

This module contains the fundamental building blocks:
- PE: Processing Element (MAC unit)
- PEArray: Combinational grid of PEs
- SystolicArray: Pipelined grid of PEArrays

Note: Tile and Mesh are deprecated aliases for PEArray and SystolicArray.
"""

from .pe import PE, PEWithShift
from .pe_array import PEArray
from .systolic_array import SystolicArray

# Deprecated aliases for backwards compatibility
Tile = PEArray
Mesh = SystolicArray

__all__ = ["PE", "PEWithShift", "PEArray", "SystolicArray", "Tile", "Mesh"]
