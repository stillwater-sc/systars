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

## Matmul Example

The animation shows:

  1. 4×4 PE Array executing Cannon's algorithm for matrix multiplication
  2. Each cycle displays:
    - Current instruction (IMUL, IADD, XNET_E, XNET_S)
    - Stall cycles for multi-cycle operations (IMUL takes 4 cycles)
    - Register A (blue), B (magenta), and C (accumulator, green when matching expected)
  3. XNET shifts: A shifts west, B shifts north with toroidal wrap-around
  4. Final statistics: 28 cycles, 16 instructions (4 iterations × 4 instructions each)

You can run it interactively with:

```bash
  python examples/simd/maspar/01_animated_maspar.py --step   # Step through manually
  python examples/simd/maspar/01_animated_maspar.py          # 500ms delay animation
  python examples/simd/maspar/01_animated_maspar.py --m 8    # 8×8 array
```

## Conv2D Example

The Conv2D implementation is complete. Here's a summary:

Added to maspar_sim.py:

- load_conv2d_data(image, kernel) - Load image and 3x3 kernel
- create_conv2d_program() - Generate convolution instructions:
    a. Load 9 kernel weights via LDI (9 instructions)
    b. Gather 8 neighbors via XNET (8 instructions)
    c. Initialize output to zero (1 instruction)
    d. 9 multiply-accumulate operations (18 instructions)
- run_conv2d(image, kernel) - Execute complete convolution
- extract_conv2d_result() - Get output from PE registers

New animation script: examples/simd/maspar/02_animated_conv2d.py

- Supports multiple kernels: identity, box, edge, sharpen, sobel_x, sobel_y
- Shows XNET neighbor gathering and MAC accumulation
- Verifies results against reference implementation

Usage:

```bash
  python examples/simd/maspar/02_animated_conv2d.py                    # 8x8 edge detection
  python examples/simd/maspar/02_animated_conv2d.py --kernel box       # Box blur
  python examples/simd/maspar/02_animated_conv2d.py --kernel sharpen   # Sharpen
  python examples/simd/maspar/02_animated_conv2d.py --size 16          # 16x16 image
```

Performance: 36 instructions, 63 cycles (due to 4-cycle IMUL latency) for any image size - the beauty of SIMD parallelism!
