"""
SIMT (Single Instruction Multiple Thread) implementations.

This package provides SIMT architecture implementations for comparing
energy efficiency against systolic arrays.

Architectures:
- **nv**: NVIDIA Streaming Multiprocessor (SM) - warp-based SIMT
- **amd**: AMD RDNA/CDNA Workgroup Processor (WGP) - wave-based SIMT [future]

For classic SIMD (lock-step, no divergence), see systars.simd.

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

Architecture Comparison:
    | Aspect          | NVIDIA SM       | AMD WGP        |
    |-----------------|-----------------|----------------|
    | Model           | SIMT (warp)     | SIMT (wave)    |
    | Width           | 32 threads      | 32/64 threads  |
    | Divergence      | Predication     | Predication    |
    | Local Memory    | Shared (48KB)   | LDS (64KB)     |
"""

# Re-export NVIDIA implementation for backward compatibility
# This allows `from systars.simt import SMSim` to work as before
from .nv import *  # noqa: F401, F403

# Re-export __all__ from nv for proper module introspection
from .nv import __all__  # noqa: F401
