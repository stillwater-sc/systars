"""
ISA Instructions for the SYSTARS accelerator.

This module provides high-level ISA instructions that automatically handle
tiling, scheduling, and command generation for operations that exceed the
systolic array dimensions.

Instructions:
    Matmul: Matrix multiply C = A @ B + D
    Conv2d: 2D convolution Y = conv(X, F) + B

The hardware automatically:
- Selects optimal dataflow (output-stationary, A-stationary, B-stationary)
- Tiles computations to fit array dimensions
- Manages double buffering for latency hiding
- Generates internal command sequences (LOAD, PRELOAD, COMPUTE, STORE)
"""

from .matmul import Matmul

__all__ = [
    "Matmul",
]
