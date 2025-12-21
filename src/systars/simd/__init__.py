"""
SIMD (Single Instruction Multiple Data) implementations.

This package provides classic SIMD array processor implementations for
comparing energy efficiency against systolic arrays and SIMT architectures.

Architectures:
- **maspar**: MasPar MP-1/MP-2 SIMD array processor

Key Differences from SIMT (NVIDIA/AMD):
- Pure lock-step execution - no divergence handling
- Single instruction stream for all processing elements
- 2D mesh communication patterns
- Designed for data-parallel array operations

Usage:
    from systars.simd.maspar import MasParSim, MasParConfig

    config = MasParConfig(array_rows=4, array_cols=4)
    sim = MasParSim(config)
    cycles = sim.run_gemm(A, B)
    C = sim.extract_result(rows=4, cols=4)

See Also:
    - systars.simt: SIMT implementations (NVIDIA, AMD)
    - systars.core: Systolic array implementations
"""

# Re-export MasPar implementation
from .maspar import *  # noqa: F401, F403
from .maspar import __all__  # noqa: F401
