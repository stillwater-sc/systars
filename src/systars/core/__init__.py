"""
Core systolic matmul array components.

This module contains the fundamental building blocks:
- PE: Processing Element (MAC unit, output- or weight-stationary)
- PEArray: Combinational grid of PEs
- SystolicArray: Pipelined grid of PEArrays, with input skew and output deskew
"""

from .pe import OUTPUT_STATIONARY, PE, WEIGHT_STATIONARY
from .pe_array import PEArray
from .systolic_array import SystolicArray

__all__ = ["OUTPUT_STATIONARY", "PE", "PEArray", "SystolicArray", "WEIGHT_STATIONARY"]
