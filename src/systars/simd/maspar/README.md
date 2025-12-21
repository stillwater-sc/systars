# MasPar MP-1 and MP-2 designs

The MasPar SIMD array processor implementation is complete. Here's a summary of what was created:

Files Created:

  | File                                       | Purpose                                                                                          |
  |--------------------------------------------|--------------------------------------------------------------------------------------------------|
  | src/systars/simd/maspar/config.py          | MasParConfig dataclass with hardware parameters (array size, registers, latencies, energy model) |
  | src/systars/simd/maspar/instruction.py     | Opcode enum (40+ opcodes), XNETDirection, Instruction dataclass                                  |
  | src/systars/simd/maspar/pe.py              | Processing Element with 64 registers, local memory, ALU execute                                  |
  | src/systars/simd/maspar/pe_array.py        | 2D mesh with XNET 8-neighbor connectivity (toroidal)                                             |
  | src/systars/simd/maspar/acu.py             | Array Control Unit for instruction fetch/decode                                                  |
  | src/systars/simd/maspar/maspar_sim.py      | Top-level simulator with GEMM support (Cannon's algorithm)                                       |
  | src/systars/simd/maspar/__init__.py        | Package exports                                                                                  |
  | examples/simd/maspar/01_animated_maspar.py | Animated GEMM visualization                                                                      |

Key Features:

- Pure SIMD execution (no divergence handling like SIMT)
- Cannon's algorithm for matrix multiplication with pre-skewed data distribution
- XNET mesh with 8 directions (N, S, E, W, NE, NW, SE, SW) and toroidal wrap-around
- Multi-cycle instruction support (IMUL = 4 cycles, FPU = 4 cycles)
- PE masking for conditional execution via predicates

Usage:

```python
  from systars.simd.maspar import MasParSim, MasParConfig
  import numpy as np

  config = MasParConfig(array_rows=4, array_cols=4)
  sim = MasParSim(config)
  cycles = sim.run_gemm(A, B)
  C = sim.extract_result(rows=4, cols=4)
```
