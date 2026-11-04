"""
Core systolic matmul array components.

This module contains the fundamental building blocks:
- PE: Processing Element (MAC unit with a register on every link; OS or stationary)
- PEArray: Combinational grid of PEs
- SystolicArray: Grid of PEArrays; the L1 stream buffers stagger its streams
"""

from .pe import OUTPUT_STATIONARY, PE, WEIGHT_STATIONARY
from .pe_array import PEArray
from .systolic_array import SystolicArray

__all__ = ["OUTPUT_STATIONARY", "PE", "PEArray", "SystolicArray", "WEIGHT_STATIONARY"]
