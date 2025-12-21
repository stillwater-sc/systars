# Session Log: MasPar MP-2 SIMD Implementation

**Date:** 2025-12-21
**Focus:** Implement MasPar MP-1/MP-2 SIMD array processor with GEMM and Conv2D support

## Summary

Implemented a complete MasPar MP-2 SIMD array processor simulator capable of executing matrix multiplication (using Cannon's algorithm) and 2D convolution with 3x3 kernels. The implementation leverages the MasPar's 8-neighbor XNET mesh for efficient data movement.

## Background

The MasPar MP-1/MP-2 was a classic SIMD machine from the late 1980s/early 1990s. Key characteristics:

- **Pure SIMD**: All PEs execute the same instruction in lockstep (no divergence handling)
- **2D Mesh**: PEs arranged in a 2D grid (typically 64x64 = 4096 PEs)
- **XNET**: 8-neighbor mesh connectivity (N, S, E, W, NE, NW, SE, SW) with toroidal wrap-around
- **ACU**: Array Control Unit broadcasts instructions to all PEs
- **Historical Significance**: MasPar architects later designed NVIDIA's GPU architecture

## Changes Made

### 1. Directory Reorganization

Moved MasPar from `simt/` to new `simd/` package (MasPar is pure SIMD, not SIMT):

```
src/systars/simd/
├── __init__.py           # Re-exports maspar
└── maspar/
    ├── __init__.py       # Package exports
    ├── config.py         # MasParConfig dataclass
    ├── instruction.py    # Opcode enum, Instruction dataclass
    ├── pe.py             # Processing Element
    ├── pe_array.py       # 2D mesh with XNET
    ├── acu.py            # Array Control Unit
    └── maspar_sim.py     # Top-level simulator
```

### 2. Core Components Implemented

**MasParConfig** (`config.py`):

- Array dimensions (default 64x64)
- PE specs: 64 registers, 32-bit ALU, 64KB local memory
- XNET latency (1 cycle) and router latency (10 cycles)
- Energy model parameters
- Pre-defined configs: DEFAULT, SMALL (4x4), MEDIUM (16x16), LARGE (128x128), MP1

**Opcode Enum** (`instruction.py`):

- Integer ALU: IADD, ISUB, IMUL, IAND, IOR, IXOR, ISHL, ISHR, IMOV
- Floating-point: FADD, FSUB, FMUL, FFMA
- Memory: LD, ST, LDI
- XNET: XNET_N/S/E/W/NE/NW/SE/SW (8 directions)
- Router: ROUTE_SEND, ROUTE_RECV
- Control: SETMASK, CLRMASK
- Reduction: REDUCE_SUM/MAX/MIN
- Comparison: ICMP_EQ/NE/LT/LE/GT/GE

**Processing Element** (`pe.py`):

- 64 x 32-bit registers (numpy int32 array)
- 64KB local memory (byte array)
- Active/predicate state for masking
- Execute method for all opcodes
- IEEE 754 float support via bit casting

**PE Array** (`pe_array.py`):

- 2D grid of PEs with row/col indexing
- `broadcast_instruction()`: SIMD dispatch
- `xnet_shift()`: Toroidal neighbor communication
- `set_mask_by_predicate()`: Conditional execution
- `load_matrix_block()` / `extract_matrix_block()`: Data loading/extraction

**Array Control Unit** (`acu.py`):

- Program storage and program counter
- Multi-cycle instruction stalling (IMUL = 4 cycles)
- Scalar registers for loop control

**MasParSim** (`maspar_sim.py`):

- Coordinates ACU and PE array
- Statistics collection
- GEMM support with Cannon's algorithm
- Conv2D support with 3x3 kernel

### 3. Matrix Multiplication (GEMM)

Implemented Cannon's algorithm for parallel matrix multiplication:

1. **Pre-skew**: Data distributed with row/column skew
   - PE(i,j) gets A[i, (i+j) mod K] and B[(i+j) mod K, j]
2. **Iteration loop** (K times):
   - IMUL: C += A * B (local MAC)
   - XNET_E: Shift A west (read from east neighbor)
   - XNET_S: Shift B north (read from south neighbor)
3. **Result**: Each PE holds its output element

Performance: 4x4 GEMM = 29 cycles (16 instructions with 4-cycle IMUL)

### 4. 2D Convolution (Conv2D)

Implemented 3x3 convolution leveraging XNET's 8-neighbor mesh:

1. **Load kernel weights** (9 LDI instructions)
2. **Gather neighbors** (8 XNET instructions for all directions)
3. **Compute MAC** (9 multiply-accumulate pairs = 18 instructions)

Total: 36 instructions, 63 cycles for any image size.

Supported kernels: identity, box, edge, sharpen, sobel_x, sobel_y

### 5. Animation Examples

**01_animated_maspar.py**: GEMM visualization

- Shows PE array with A, B, C registers
- Displays XNET shift directions
- Tracks iteration progress
- Verifies result against numpy

**02_animated_conv2d.py**: Conv2D visualization

- Shows input image and output accumulator
- Displays XNET neighbor gathering
- Supports multiple kernel types
- Phase tracking: Load Kernel → XNET Gather → MAC

### 6. CLI Improvements

Fixed animation jumpiness by ensuring consistent layout:

- XNET status line always prints (blank if not active)
- Added separator line between XNET message and PE arrays

## Files Created

| File | Purpose |
|------|---------|
| `src/systars/simd/__init__.py` | SIMD package with MasPar re-exports |
| `src/systars/simd/maspar/__init__.py` | MasPar exports (16 symbols) |
| `src/systars/simd/maspar/config.py` | MasParConfig + pre-defined configs |
| `src/systars/simd/maspar/instruction.py` | 40+ opcodes, Instruction dataclass |
| `src/systars/simd/maspar/pe.py` | Processing Element with ALU |
| `src/systars/simd/maspar/pe_array.py` | 2D mesh with XNET |
| `src/systars/simd/maspar/acu.py` | Array Control Unit |
| `src/systars/simd/maspar/maspar_sim.py` | Top-level simulator |
| `examples/simd/maspar/01_animated_maspar.py` | GEMM animation |
| `examples/simd/maspar/02_animated_conv2d.py` | Conv2D animation |

## Files Modified

| File | Changes |
|------|---------|
| `src/systars/simt/__init__.py` | Removed maspar reference, pointed to simd |

## Test Results

All 328 unit tests pass:

```
============================= 328 passed in 19.70s =============================
```

GEMM verification:

- 4x4 identity: PASS
- 4x4 random: PASS

Conv2D verification:

- Identity kernel: PASS
- Box blur (interior): PASS
- Edge detection (interior): PASS

## Usage Examples

```python
from systars.simd.maspar import MasParSim, MasParConfig
import numpy as np

# GEMM
config = MasParConfig(array_rows=4, array_cols=4)
sim = MasParSim(config)
A = np.random.randint(1, 10, (4, 4), dtype=np.int32)
B = np.random.randint(1, 10, (4, 4), dtype=np.int32)
cycles = sim.run_gemm(A, B)
C = sim.extract_result(rows=4, cols=4)

# Conv2D
edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.int32)
cycles = sim.run_conv2d(image, edge_kernel)
output = sim.extract_conv2d_result(rows=8, cols=8)
```

## Architecture Comparison

| Aspect | NVIDIA SM | MasPar MP-2 |
|--------|-----------|-------------|
| Model | SIMT (warp) | Pure SIMD |
| Width | 32 threads | 4096+ PEs |
| Divergence | Yes (predication) | No (lock-step) |
| Local Mem | Shared 48KB | Per-PE 64KB |
| Connectivity | Shuffle | 8-neighbor mesh + router |
| Communication | Warp shuffle | XNET (1 cycle) |

## Future Work

1. **Global Router**: Implement arbitrary PE-to-PE routing
2. **PE Masking**: Add SETMASK/CLRMASK for conditional execution
3. **Larger Kernels**: Extend Conv2D to 5x5, 7x7 kernels
4. **Energy Model**: Add energy tracking like SIMT implementation
5. **Comparison Study**: Benchmark SIMD vs SIMT on same workloads
