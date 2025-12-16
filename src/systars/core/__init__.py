"""
Core systolic matmul array components.

This module contains the fundamental building blocks:
- PE: Processing Element (MAC unit)
- PEArray: Combinational grid of PEs
- SystolicArray: Pipelined grid of PEArrays
"""

from .pe import PE, PEWithShift
from .pe_array import PEArray
from .systolic_array import SystolicArray

__all__ = ["PE", "PEWithShift", "PEArray", "SystolicArray"]
