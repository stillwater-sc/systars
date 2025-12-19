"""
Stencil Machine for SYSTARS accelerator.

This module provides a dedicated stencil machine for energy-efficient
2D convolution and spatial operations. It uses line buffers for optimal
input data reuse, achieving 1x DRAM reads per input pixel (vs 9x for
im2col with 3x3 kernels).

Components:
    StencilConfig: Configuration dataclass for stencil machine
    LineBufferBank: Single line buffer SRAM bank
    LineBufferUnit: Collection of K_h line buffers
    WindowFormer: Sliding window extraction via shift registers
    MACBank: Single output channel MAC unit
    ChannelParallelMAC: Parallel MAC array for P_c output channels
    StencilMachine: Top-level integration

The stencil machine is designed to work alongside the systolic array:
    - Systolic Array: Pure GEMM operations (FC layers, attention)
    - Stencil Machine: Conv2D, depthwise conv, pooling
"""

from .config import StencilActivation, StencilConfig, StencilDataflow
from .line_buffer import LineBufferBank, LineBufferUnit
from .mac_array import ChannelParallelMAC, MACBank
from .stencil_machine import StencilMachine, StencilState
from .window_former import WindowFormer

__all__ = [
    "StencilConfig",
    "StencilActivation",
    "StencilDataflow",
    "LineBufferBank",
    "LineBufferUnit",
    "WindowFormer",
    "MACBank",
    "ChannelParallelMAC",
    "StencilMachine",
    "StencilState",
]
