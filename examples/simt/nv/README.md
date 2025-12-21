# NVIDIA Streaming Multiprocessor Examples

Examples for NVIDIA SM-style SIMT simulation with SM-level LSU.

## Examples

- **01_animated_simt.py**: Animated terminal visualization of SM execution
  - Shows 4 partitions with warp schedulers, ALUs, and memory operations
  - Tiled GEMM with round-robin warp distribution
  - MSHR-based memory coalescing

- **02_energy_comparison.py**: Energy efficiency comparison
  - SIMT vs Systolic Array vs Stencil Machine
  - Shows instruction overhead in SIMT (~90% non-compute energy)

- **03_gemm_functional.py**: Functional GEMM simulation
  - Validates matrix multiply correctness
  - Uses simplified execution model

## Usage

```bash
# Animated SIMT visualization
python examples/simt/nv/01_animated_simt.py --size 8 --warps 4

# Energy comparison
python examples/simt/nv/02_energy_comparison.py --size 64 --detailed
```
