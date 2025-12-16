"""
Common utilities for systars examples.

This module provides shared infrastructure for demonstration applications,
including simulated memory models and tensor manipulation utilities.
"""

from .dram_model import SimulatedDRAM
from .tensor_utils import (
    pack_matrix_int8,
    pack_matrix_int32,
    tile_matrix,
    unpack_matrix_int8,
    unpack_matrix_int32,
    untile_matrix,
)

__all__ = [
    "SimulatedDRAM",
    "pack_matrix_int8",
    "unpack_matrix_int8",
    "pack_matrix_int32",
    "unpack_matrix_int32",
    "tile_matrix",
    "untile_matrix",
]
