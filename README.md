# SYSTARS: A modern EDA tool for the generation, analysis, and synthesis of Systolic/Wavefront arrays

A Python-based systolic array RTL generator using [Amaranth HDL](https://amaranth-lang.org/).

Systars generates synthesizable Verilog for configurable systolic array accelerators, targeting linear algebra workloads in signal processing, sensor fusion, and real-time control. It is a modern Python reimplementation of the original SYSTARS work at Delft University of Technology, which synthesised systolic and wavefront arrays from recurrence equations (see [Lineage and credits](#lineage-and-credits)).

[SYSTARS: A CAD tool for the synthesis and analysis of VLSI systolic/wavefront arrays](https://ieeexplore.ieee.org/document/18078)

```bibtex
@INPROCEEDINGS{18078,
  author={Omtzigt, E.T.L.},
  booktitle={[1988] Proceedings. International Conference on Systolic Arrays},
  title={SYSTARS: A CAD tool for the synthesis and analysis of VLSI systolic/wavefront arrays},
  year={1988},
  volume={},
  number={},
  pages={383-391},
  keywords={Very large scale integration;Algorithm design and analysis;Systolic arrays;Clustering algorithms;Partitioning algorithms;Design automation;Iterative algorithms;Animation;Computer graphics;Adaptive arrays},
  doi={10.1109/ARRAYS.1988.18078}}
```

## Features

- **Configurable Systolic Array**: Parameterized grid dimensions, data types, and dataflows
- **Universal Dataflow Support**: Output-stationary (OS), input-stationary (IS), and weight-stationary (WS) modes
- **Modern Python**: Clean, type-hinted Python 3.10+ codebase
- **Amaranth HDL**: Leverage Python's expressiveness for hardware design
- **Comprehensive Testing**: Unit tests with Amaranth's built-in simulator
- **A KPU compute tile**: `SystolicTop` moves whole tiles between L3, L2 and L1 on credits and tile-ready tokens

New to the project? Start with the [introduction](docs/getting-started/introduction.md) and
[your first week](docs/getting-started/onboarding.md), or the
[documentation site](https://stillwater-sc.github.io/systars/).

## Installation

### Full Development Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/stillwater-sc/systars.git
cd systars

# Run the setup script (downloads OSS-CAD-SUITE, creates venv, installs deps)
./bin/setup-dev-env.sh

# Activate in a new shell:
source .venv/bin/activate
```

### Manual Setup

```bash
# Set OSS-CAD-SUITE location (if not in /opt/oss-cad-suite)
export OSS_CAD_SUITE=/path/to/oss-cad-suite

# Use just to create venv and install dependencies
just setup

# Activate:
source .venv/bin/activate
```

### Requirements

- Python 3.10+
- [OSS-CAD-SUITE](https://github.com/YosysHQ/oss-cad-suite-build) for EDA tools:
  - Verilator (RTL simulation)
  - Yosys (synthesis)
  - GHDL (VHDL simulation)
- [just](https://github.com/casey/just) task runner (optional but recommended)

## Quick Start

```python
from systars.config import SystolicConfig, Dataflow
from systars.core.pe import PE
from amaranth.back import verilog

# Create a configuration
config = SystolicConfig(
    grid_rows=16,
    grid_cols=16,
    input_bits=8,
    acc_bits=32,
    dataflow=Dataflow.OUTPUT_STATIONARY | Dataflow.B_STATIONARY,
)

# Generate a Processing Element
pe = PE(config)

# Generate Verilog
v = verilog.convert(pe, name="PE")
print(v)
```

## Architecture

```
┌────────────────────────────────────────────┐
│             Systolic Array                 │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐           │
│  │ PE  │→│ PE  │→│ PE  │→│ PE  │ ...       │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘           │
│     ↓       ↓       ↓       ↓              │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐           │
│  │ PE  │→│ PE  │→│ PE  │→│ PE  │ ...       │
│  └─────┘ └─────┘ └─────┘ └─────┘           │
└────────────────────────────────────────────┘
```

### Components

| Component | Status | Description |
|-----------|--------|-------------|
| PE | ✅ Done | Processing Element with MAC |
| PEArray | ✅ Done | Combinational PE grid (tile_rows × tile_cols) |
| SystolicArray | ✅ Done | Pipelined PEArray grid (grid_rows × grid_cols) |
| Scratchpad | ✅ Done | Multi-bank local memory |
| Combiner (storage) | ✅ Done | Result memory with ReLU/ReLU6 on readout (no scaling yet); the [Combiner](docs/architecture/combiner.md)'s function is #38 |
| Streamer | ✅ Done | L2 → L1 → fabric: EXECUTE commands (CONFIG, PRELOAD, COMPUTE) |
| BlockMoverIn | ✅ Done | The BlockMover's inbound leg, L3 → L2 (LOAD) |
| BlockMoverOut | ✅ Done | The BlockMover's outbound leg, Combiner → L3 (STORE) |
| StreamReader | ✅ Done | AXI-like burst read engine of the BlockMover |
| StreamWriter | ✅ Done | AXI-like burst write engine of the BlockMover |
| DescriptorEngine | ✅ Done | BlockMover descriptor chain execution |
| LoopMatmul | 🔲 TODO | High-level matmul loop unroller |
| LoopConv | 🔲 TODO | Convolution loop unroller |

## Configuration

Key parameters in `SystolicConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_rows` | 16 | SystolicArray height (number of PEArrays) |
| `grid_cols` | 16 | SystolicArray width (number of PEArrays) |
| `tile_rows` | 1 | PEArray height (PEs per tile) |
| `tile_cols` | 1 | PEArray width (PEs per tile) |
| `input_bits` | 8 | Input element width |
| `acc_bits` | 32 | MAC accumulate width |
| `dataflow` | OUTPUT_STATIONARY \| B_STATIONARY | Supported dataflow modes (OS and WS; IS needs `A_STATIONARY`) |
| `sp_capacity_kb` | 256 | Scratchpad size |
| `combiner_capacity_kb` | 64 | Combiner storage size |

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=systars

# Run specific test
pytest tests/unit/test_pe.py -v
```

## Generating Verilog

```python
from amaranth.back import verilog
from systars.config import SystolicConfig
from systars.core.pe import PE

config = SystolicConfig()
pe = PE(config)

# Generate Verilog
with open("pe.v", "w") as f:
    f.write(verilog.convert(pe, name="PE"))
```

## Examples & Demos

The `examples/` directory contains demonstrations of systolic array operation:

```bash
# Simple matrix multiply demo - shows command sequence and data flow
python examples/gemm/01_simple_matmul.py

# Animated wavefront visualization - terminal animation of systolic data flow
python examples/gemm/02_animated_wavefront.py --m 4 --n 4 --k 4 --delay 500

# Generate animated GIF of wavefront propagation
python examples/gemm/03_wavefront_gif.py

# Memory subsystem and dataflow architecture visualization
python examples/gemm/04_memory_dataflow.py --diagram  # Static architecture
python examples/gemm/04_memory_dataflow.py            # Animated dataflow
```

### Architecture Overview

The complete system architecture showing data flow through the memory subsystem:

```
┌────────────────────────────────────────────────────────────────────┐
│                        DRAM (External Memory)                      │
└───────────────────┬────────────────────────────────┬───────────────┘
                    │ Load                           │ Store
                    ↓                                ↑
┌──────────────────────────────────┐  ┌─────────────────────────────────┐
│         SCRATCHPAD MEMORY        │  │         COMBINER MEMORY         │
│  ┌──────┐ ┌──────┐ ┌──────┐      │  │  ┌──────┐ ┌──────┐ ┌──────┐     │
│  │Bank 0│ │Bank 1│ │Bank 2│ ...  │  │  │Bank 0│ │Bank 1│ │Bank 2│ ... │
│  │ (A)  │ │ (B)  │ │      │      │  │  │ (C)  │ │      │ │      │     │
│  └──┬───┘ └──┬───┘ └──────┘      │  │  └──────┘ └──────┘ └──────┘     │
└─────┼────────┼───────────────────┘  └─────────────────┬───────────────┘
      │        │                                        ↑
 A row│   B col│                                   C row│
stream↓  stream↓                                  stream│
    ┌────────────────────────────────────────────────────────┐
    │                    SYSTOLIC ARRAY                      │
    │   A[0]→ ┌────┐  ┌────┐  ┌────┐  ┌────┐                 │
    │         │ PE │─→│ PE │─→│ PE │─→│ PE │ → drain         │
    │   A[1]→ └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘                 │
    │           ↓       ↓       ↓       ↓                    │
    │         ┌────┐  ┌────┐  ┌────┐  ┌────┐    → C[row]     │
    │         │ PE │─→│ PE │─→│ PE │─→│ PE │                 │
    │         └────┘  └────┘  └────┘  └────┘                 │
    │          B[0]↑  B[1]↑   B[2]↑   B[3]↑                  │
    └────────────────────────────────────────────────────────┘
```

![Systolic Array Wavefront Animation](examples/animated_systolic_matmul.svg)

## Project Structure

```
systars/
├── src/systars/
│   ├── config.py          # SystolicConfig and presets
│   ├── targets.py         # The RTL targets that just gen / just synth / CI read
│   ├── top.py             # SystolicTop, the compute tile
│   ├── core/              # PE, PEArray, SystolicArray
│   ├── memory/            # Scratchpad (L2), Combiner storage, L1 stream/result buffers
│   ├── controller/        # Streamer (execute), BlockMover (load, store), credits (sync)
│   ├── blockmover/        # The BlockMover's AXI engines (StreamReader/StreamWriter)
│   ├── isa/               # Matmul and Conv2d instructions, the command port
│   ├── wavefront.py       # RTL wavefront tracing, checking and animations
│   ├── util/              # Opcodes and command helpers
│   ├── stencil/           # Side package: line-buffer stencil machine
│   ├── simt/, simd/       # Side packages: SIMT (NVIDIA-style SM) and MasPar SIMD models
│   └── loop/              # Placeholder for the loop unrollers
├── tests/unit/            # pytest + Amaranth simulator
├── verif/cocotb/          # RTL tests on Verilator
├── examples/              # Demos (gemm/, conv/, simt/, simd/, stencil/)
├── scripts/               # RTL and wavefront generation
├── docs/                  # Documentation (the source of the site)
└── docs-site/             # Starlight site built from docs/ (GitHub Pages)
```

## Roadmap

See [implementation plan](./docs/plan/implementation.md) for the detailed implementation plan.

### Phase 1: Foundation ✅

- [x] Configuration system
- [x] Processing Element (PE)
- [x] PEArray
- [x] SystolicArray

### Phase 2: Memory System ✅

- [x] Local address encoding
- [x] Scratchpad banks
- [x] Accumulator with scale/activation

### Phase 3: Controllers ✅

- [x] Execute controller (CONFIG, PRELOAD, COMPUTE)
- [x] Load controller (DRAM → Scratchpad)
- [x] Store controller (Accumulator → DRAM)

### Phase 4: DMA Engines ✅

- [x] StreamReader (AXI-like burst read)
- [x] StreamWriter (AXI-like burst write)
- [x] DescriptorEngine (DMA descriptor chains)

### Phase 5: ISA Instructions and the KPU Tile (Current)

- [x] Registered fabric, L1 stream buffers and a bubble-free Streamer (OS, WS, IS)
- [x] Credits and tile-ready tokens between the CSP processes
- [x] Matmul and Conv2d instructions, with bias and accumulate
- [x] cocotb: Matmul and Conv2d on the Verilated SystolicTop
- [ ] Banked L2 sustaining two vectors per clock (#17)
- [ ] Conv2d in WS and IS
- [ ] Hardware loop unrollers (LoopMatmul, LoopConv)

### Phase 6: Validation

- [ ] Header file generation
- [ ] Reference comparison

## License

BSD-3-Clause

## Lineage and credits

SYSTARS synthesises arrays by the space-time mapping of recurrence equations: a schedule
vector τ decides when each operation runs and a projection P decides where. The three
dataflows systars builds today (OS, WS, IS) are three projections of matrix multiply under
one linear schedule. [Lineage](docs/architecture/lineage.md) sets out the method, the
schedules beyond it that the KPU executes, and full references.

### Foundations

- R. M. Karp, R. E. Miller, S. Winograd: systems of uniform recurrence equations (*J. ACM*, 1967)
- H. T. Kung, C. E. Leiserson: systolic arrays (*Sparse Matrix Proceedings*, 1978)
- D. I. Moldovan; P. Quinton; S. Y. Kung: space-time mapping and array synthesis (1983–1988)
- E. T. L. Omtzigt: [SYSTARS](https://ieeexplore.ieee.org/document/18078) (1988), the original
  C tool at Delft University of Technology, and domain flow architectures
- [branes-ai/domain_flow](https://branes-ai.github.io/domain_flow/): SURE specifications,
  schedule legality, free and piecewise-linear schedules

### Engineering

- [Gemmini](https://github.com/ucb-bar/gemmini) (UC Berkeley; H. Genc et al., DAC 2021):
  systars began as a Python plan modelled on it, and keeps its execute command set, local
  address encoding and accumulator. See [NOTICE](NOTICE).
- [kpu-sim](https://github.com/stillwater-sc/kpu-sim): the CSP data movement model the tile
  follows
- [Amaranth HDL](https://amaranth-lang.org/): the Python HDL framework

To cite systars, use [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository").
