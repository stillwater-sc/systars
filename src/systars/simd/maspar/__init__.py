"""
MasPar MP-1/MP-2 SIMD array processor implementation.

This module provides a MasPar-style SIMD array processor implementation
based on the MP-1 and MP-2 architectures from the late 1980s/early 1990s.

Architecture:
- Array Control Unit (ACU) for instruction fetch/decode
- 2D mesh of Processing Elements (PEs) - typically 64x64 = 4096 PEs
- Pure SIMD execution (all PEs execute same instruction)
- 8-neighbor mesh connectivity (XNET) + global router
- Per-PE local RAM (64KB in MP-2)

Key Differences from NVIDIA/AMD SIMT:
- NO divergence handling - all PEs execute in lockstep
- Single instruction stream from ACU (not per-warp/wave)
- 2D mesh communication pattern (8 neighbors)
- Global router for arbitrary PE-to-PE communication
- 4-bit PEs in MP-1, 32-bit in MP-2

Historical Significance:
- MasPar was founded by Jeff Kalb in 1987
- Architects from MasPar later designed NVIDIA's GPU architecture
- The SIMT model evolved from classic SIMD to handle divergence

Usage:
    from systars.simd.maspar import MasParSim, MasParConfig
    import numpy as np

    # Create 4x4 simulator for testing
    config = MasParConfig(array_rows=4, array_cols=4)
    sim = MasParSim(config)

    # Run GEMM: C = A @ B
    A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    B = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    cycles = sim.run_gemm(A, B)
    C = sim.extract_result(rows=4, cols=4)
    print(f"Completed in {cycles} cycles")
    print(f"Result:\\n{C}")

References:
- MasPar MP-1 Architecture: https://www.cpushack.com/2014/09/05/maspar-massively-parallel-computers-32-cores-on-a-chip/
- MasPar Wikipedia: https://en.wikipedia.org/wiki/MasPar
"""

from .acu import ACU
from .config import (
    DEFAULT_MASPAR_CONFIG,
    LARGE_MASPAR_CONFIG,
    MEDIUM_MASPAR_CONFIG,
    MP1_CONFIG,
    SMALL_MASPAR_CONFIG,
    MasParConfig,
)
from .instruction import (
    OPCODE_LATENCY,
    XNET_OFFSETS,
    XNET_OPCODES,
    Instruction,
    Opcode,
    XNETDirection,
)
from .maspar_sim import MasParSim
from .pe import PE
from .pe_array import PEArray

__all__ = [
    # Simulator
    "MasParSim",
    # Components
    "ACU",
    "PE",
    "PEArray",
    # Configuration
    "MasParConfig",
    "DEFAULT_MASPAR_CONFIG",
    "SMALL_MASPAR_CONFIG",
    "MEDIUM_MASPAR_CONFIG",
    "LARGE_MASPAR_CONFIG",
    "MP1_CONFIG",
    # Instructions
    "Instruction",
    "Opcode",
    "XNETDirection",
    "XNET_OFFSETS",
    "XNET_OPCODES",
    "OPCODE_LATENCY",
]
