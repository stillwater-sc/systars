# Single Instruction Multiple Thread (SIMT) architectures

This directory hierarchy contains different SIMT micro-architecture implementations.

The current structure is:

```text
  Package (src/systars/simt/):
  simt/
  ├── __init__.py      # Compatibility layer (re-exports nv/*)
  ├── base.py          # Common protocols (ProcessorSim, MemorySubsystem)
  ├── nv/              # NVIDIA-style SM with aggregating LSU with Miss Status Handling Registers (MSHR) (14 modules)
  ├── nv_v1/           # NVIDIA-style SM with poor LSU design that limits thread request concurrency to memory (13 modules) (pedagogical example)
  ├── amd/             # AMD placeholder
  └── maspar/          # MasPar placeholder

  Examples (examples/simt/):
  simt/
  ├── nv/              # NVIDIA examples
  │   ├── 01_animated_simt.py
  │   ├── 02_energy_comparison.py
  │   ├── 03_gemm_functional.py
  │   └── README.md
  ├── nv_v1/           # Legacy examples
  │   ├── 01_animated_simt.py
  │   └── README.md
  ├── amd/             # AMD placeholder
  │   └── README.md
  └── maspar/          # MasPar placeholder
      └── README.md
```

## Note on SIMT base package

On keeping base.py at the package root: This is the idiomatic Python pattern. Common base classes
and protocols naturally belong at the package level where they're shared by all implementations.
Moving it to a base/ subpackage would add unnecessary nesting for what is (and should remain)
a minimal interface definition. If the file grows significantly in the future, we can reconsider.
