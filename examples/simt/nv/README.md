# NVIDIA Streaming Multiprocessor Examples

Examples for NVIDIA SM-style SIMT simulation with SM-level LSU.

## Terminology: "Shared Memory"

In NVIDIA GPUs, "shared memory" means **shared among threads in a thread block**, not
shared among SMs. It's programmer-managed SRAM (`__shared__` in CUDA) that enables
thread cooperation. In Volta+ GPUs, shared memory and L1 cache are unified into the
same physical SRAM with a configurable split (e.g., 48KB shared + 16KB L1).

All 4 partitions within an SM access the same shared memory banks, which is why
bank conflicts matter (32 banks, 32 threads per warp).

## Examples

- **01_animated_simt.py**: Animated terminal visualization of SM execution
  - Shows 4 partitions with warp schedulers, ALUs, and memory operations
  - Tiled GEMM with round-robin warp distribution
  - MSHR-based memory coalescing
  - `--preload` mode for observing ALU pipelining without memory latency
  - `--movie` mode for clean term2svg capture (suppresses setup/summary)

- **02_energy_comparison.py**: Energy efficiency comparison
  - SIMT vs Systolic Array vs Stencil Machine
  - Shows instruction overhead in SIMT (~90% non-compute energy)

- **03_gemm_functional.py**: Functional GEMM simulation
  - Validates matrix multiply correctness
  - Uses simplified execution model

## Usage

```bash
# Animated SIMT visualization with tiled GEMM
python examples/simt/nv/01_animated_simt.py --tiled --m 8 --n 8 --k 2 --fast --fast-mem

# Preload mode: focus on register fetches and ALU pipelining
# Matrices preloaded into shared memory, eliminating memory latency
python examples/simt/nv/01_animated_simt.py --tiled --m 8 --n 4 --k 4 --preload --fast

# Movie mode: clean output for term2svg capture (no setup/summary)
python examples/simt/nv/01_animated_simt.py --tiled --m 8 --n 8 --k 2 --fast --movie

# Create term2svg animation
term2svg -g 100x50 -c "python examples/simt/nv/01_animated_simt.py --tiled --m 8 --n 8 --k 2 --movie" simt_demo.svg

# Energy comparison
python examples/simt/nv/02_energy_comparison.py --size 64 --detailed
```
