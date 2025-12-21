# Session Log: SIMT Package Reorganization for Multi-Architecture Support

**Date:** 2025-12-20
**Focus:** Reorganize SIMT package to support NVIDIA, AMD, and MasPar architectures

## Summary

Reorganized the SIMT package from a single NVIDIA-focused implementation to a multi-architecture structure. Moved all NVIDIA code to `simt/nv/`, created placeholders for AMD RDNA/CDNA and MasPar architectures, and added a compatibility layer for backward-compatible imports.

## Background

The existing SIMT implementation was NVIDIA-specific (Streaming Multiprocessor with warp-based SIMT). To enable comparison with other parallel architectures:

- **AMD RDNA/CDNA**: Workgroup Processor with wave-based SIMT (32 or 64 threads)
- **MasPar MP-1/MP-2**: Classic SIMD array processor with 2D mesh of Processing Elements

The MasPar architecture is historically significant as its architects later designed NVIDIA's GPU architecture.

## Changes Made

### 1. Round-Robin Warp Distribution Fix

Fixed warp distribution to use NVIDIA-style round-robin instead of sequential fill:

**Before (sequential):**

```python
partition_id = i // warps_per_partition  # Fills P0 completely before P1
# 18 warps: P0=8, P1=8, P2=2, P3=0
```

**After (round-robin):**

```python
partition_id = i % num_partitions  # Distributes evenly
# 18 warps: P0=5, P1=5, P2=4, P3=4
```

Files modified:

- `src/systars/simt/sm_controller.py`: Fixed `load_uniform_program()` and `activate_warps()`
- `examples/simt/01_animated_simt.py`: Fixed tiled GEMM warp distribution and global warp ID calculation

### 2. Package Reorganization

Created new directory structure:

```
src/systars/simt/
├── __init__.py         # Compatibility layer (re-exports nv/*)
├── base.py             # Common protocols (ProcessorSim, MemorySubsystem)
├── nv/                 # NVIDIA SM with SM-level LSU (current)
│   ├── __init__.py     # 63+ exports
│   ├── config.py
│   ├── sm_controller.py
│   ├── partition.py
│   ├── warp_scheduler.py
│   ├── register_file.py
│   ├── operand_collector.py
│   ├── execution_unit.py
│   ├── shared_memory.py
│   ├── global_memory.py
│   ├── load_store_unit.py
│   ├── memory_coalescer.py
│   ├── sm_lsu.py
│   ├── barrier.py
│   ├── energy_model.py
│   └── README.md
├── nv_v1/              # NVIDIA SM with per-partition LSU bug (legacy)
│   └── (same modules as nv/)
├── amd/                # AMD RDNA/CDNA placeholder
│   └── __init__.py     # NotImplementedError stubs
└── maspar/             # MasPar MP-1/MP-2 placeholder
    └── __init__.py     # NotImplementedError stubs
```

### 3. Backward Compatibility

The compatibility layer allows existing code to work unchanged:

```python
# These still work (defaults to NVIDIA):
from systars.simt import SMSim, SIMTConfig

# Explicit architecture selection:
from systars.simt.nv import SMSim, SIMTConfig
from systars.simt.nv_v1 import SMSim  # Legacy with LSU bug
from systars.simt.amd import WGPSim  # NotImplementedError
from systars.simt.maspar import PEArraySim  # NotImplementedError
```

### 4. Common Base Classes

Created `base.py` with protocols for architecture-agnostic interfaces:

```python
class ProcessorSim(Protocol):
    def step(self) -> dict: ...
    def run_to_completion(self, max_cycles: int) -> int: ...
    def done(self) -> bool: ...
    def get_statistics(self) -> dict: ...
    def get_energy_pj(self) -> float: ...

class MemorySubsystem(Protocol):
    def read(self, address: int) -> int: ...
    def write(self, address: int, value: int) -> None: ...

class SIMTConfigBase(ABC):
    def execution_width(self) -> int: ...  # 32/64/4096+
    def local_memory_kb(self) -> int: ...  # 48/64/16KB
    def num_compute_units(self) -> int: ...  # 4/2/1
```

## Architecture Comparison

| Aspect | NVIDIA SM | AMD WGP | MasPar MP-1/2 |
|--------|-----------|---------|---------------|
| Model | SIMT (warp) | SIMT (wave) | Pure SIMD |
| Width | 32 threads | 32/64 threads | 4096+ PEs |
| Divergence | Yes | Yes | No (lock-step) |
| Local Mem | Shared 48KB | LDS 64KB | PE RAM 16KB |
| Connectivity | Shuffle | Permute | 8-neighbor mesh + router |

## Files Modified

- `src/systars/simt/__init__.py` - Replaced with compatibility layer
- `src/systars/simt/nv/__init__.py` - Created with all exports
- `src/systars/simt/nv/*.py` - Moved from simt/
- `src/systars/simt/nv_v1/*.py` - Moved from simt_v1/
- `src/systars/simt/amd/__init__.py` - Created placeholder
- `src/systars/simt/maspar/__init__.py` - Created placeholder
- `src/systars/simt/base.py` - Created common protocols
- `tests/unit/test_simt_components.py` - Updated imports to nv/
- `examples/simt/01_animated_simt.py` - Fixed round-robin distribution
- `examples/simt/02_energy_comparison.py` - Updated imports
- `examples/simt/v1/01_animated_simt.py` - Updated to use simt.nv_v1

## Test Results

All 57 SIMT tests pass with the new structure:

- test_simt_components.py: 37 tests
- test_simt_memory.py: 20 tests

## Research Sources

- [AMD RDNA Wikipedia](https://en.wikipedia.org/wiki/RDNA_(microarchitecture))
- [AMD RDNA4 at Hot Chips 2025](https://chipsandcheese.com/p/amds-rdna4-gpu-architecture-at-hot)
- [AMD UDNA Announcement](https://www.tomshardware.com/pc-components/cpus/amd-announces-unified-udna-gpu-architecture-bringing-rdna-and-cdna-together-to-take-on-nvidias-cuda-ecosystem)
- [MasPar MP-1 Architecture](https://www.researchgate.net/publication/316514816_MasPar_MP-1_An_SIMD_Array_Processor)
- [MasPar Wikipedia](https://en.wikipedia.org/wiki/MasPar)
- [MasPar: 32 Cores on a Chip](https://www.cpushack.com/2014/09/05/maspar-massively-parallel-computers-32-cores-on-a-chip/)

## Future Work

1. **AMD Implementation**: WGPSim with wave scheduler, LDS, 32/64-thread waves
2. **MasPar Implementation**: ACU for instruction fetch, 2D PE array, 8-neighbor mesh + global router
3. **Energy Comparison**: Compare all three architectures on GEMM workloads
