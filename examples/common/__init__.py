"""
Common utilities for systars examples.

This module provides shared infrastructure for demonstration applications,
including simulated memory models, tensor manipulation utilities, and
common CLI argument definitions.
"""

from .cli import (
    AnimationArgs,
    add_animation_args,
    add_gemm_args,
    add_memory_args,
    add_timeline_args,
    get_effective_delay,
    should_print,
    should_prompt,
)
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
    # CLI utilities
    "AnimationArgs",
    "add_animation_args",
    "add_gemm_args",
    "add_memory_args",
    "add_timeline_args",
    "get_effective_delay",
    "should_print",
    "should_prompt",
    # Memory model
    "SimulatedDRAM",
    # Tensor utilities
    "pack_matrix_int8",
    "unpack_matrix_int8",
    "pack_matrix_int32",
    "unpack_matrix_int32",
    "tile_matrix",
    "untile_matrix",
]
