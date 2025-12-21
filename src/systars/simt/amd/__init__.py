"""
AMD RDNA/CDNA Workgroup Processor (WGP) implementation.

This module will provide an AMD-style Workgroup Processor implementation
based on the RDNA/CDNA architecture.

Architecture (planned):
- Workgroup Processor (WGP) containing 2 Compute Units (CUs)
- Wave-based SIMT (32 or 64 threads per wave)
- Local Data Share (LDS) - 64KB shared memory
- Vector ALUs and Scalar ALUs
- Wave scheduler with GDS (Global Data Share)

Key Differences from NVIDIA:
- Wave size can be 32 (wave32) or 64 (wave64) threads
- 2 CUs share a WGP and coordinate on LDS
- Different memory hierarchy (L0/L1/L2 vs NVIDIA's L1/L2)
- VGPR (Vector) and SGPR (Scalar) register files

References:
- AMD RDNA Architecture: https://gpuopen.com/amd-gpu-architecture-programming-documentation/
- AMD RDNA4 at Hot Chips 2025: https://chipsandcheese.com/p/amds-rdna4-gpu-architecture-at-hot

Status: Not yet implemented.
"""


def _not_implemented() -> None:
    """Raise NotImplementedError when attempting to use AMD module."""
    raise NotImplementedError(
        "AMD RDNA/CDNA Workgroup Processor implementation coming soon.\n"
        "See the NVIDIA implementation at systars.simt.nv for a working example."
    )


# Placeholder exports that raise NotImplementedError
class WGPSim:
    """AMD Workgroup Processor simulation (not yet implemented)."""

    def __init__(self, *_args, **_kwargs) -> None:
        _not_implemented()


class AMDConfig:
    """AMD WGP configuration (not yet implemented)."""

    def __init__(self, *_args, **_kwargs) -> None:
        _not_implemented()


__all__ = ["WGPSim", "AMDConfig"]
