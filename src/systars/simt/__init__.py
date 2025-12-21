"""
SIMT (Single Instruction Multiple Thread) implementations.

This package provides multiple SIMT/SIMD architecture implementations for
comparing energy efficiency against systolic arrays.

Architectures:
- **nv**: NVIDIA Streaming Multiprocessor (SM) - warp-based SIMT
- **amd**: AMD RDNA/CDNA Workgroup Processor (WGP) - wave-based SIMT [future]
- **maspar**: MasPar MP-1/MP-2 SIMD array - classic SIMD [future]

Usage (backward compatible - defaults to NVIDIA):
    from systars.simt import SIMTConfig, SMSim

    config = SIMTConfig()
    sm = SMSim(config)
    sm.load_uniform_program(program, num_warps=4)
    sm.activate_warps(4)
    sm.run_to_completion()

Explicit architecture selection:
    from systars.simt.nv import SMSim, SIMTConfig
    from systars.simt.amd import WGPSim  # future
    from systars.simt.maspar import PEArraySim  # future

Architecture Comparison:
    | Aspect          | NVIDIA SM       | AMD WGP        | MasPar MP-1/2  |
    |-----------------|-----------------|----------------|----------------|
    | Model           | SIMT (warp)     | SIMT (wave)    | Pure SIMD      |
    | Width           | 32 threads      | 32/64 threads  | 4096+ PEs      |
    | Divergence      | Yes             | Yes            | No (lock-step) |
    | Local Memory    | Shared (48KB)   | LDS (64KB)     | PE RAM (16KB)  |
"""

# Re-export NVIDIA implementation for backward compatibility
# This allows `from systars.simt import SMSim` to work as before
from .nv import *  # noqa: F401, F403

# Re-export __all__ from nv for proper module introspection
from .nv import __all__  # noqa: F401
