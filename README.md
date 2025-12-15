# Systars

A Python-based systolic array RTL generator using [Amaranth HDL](https://amaranth-lang.org/).

Systars generates synthesizable Verilog for configurable systolic array accelerators, targeting linear algebra workloads in signal processing, sensor fusion, and real-time control. It is a modern Python reimplementation inspired by the original SYSTARS work at Delft University Of Technology.

[SYSTARS: A CAD tool for the synthesis and analysis of VLSI systolic/wavefront arrays](https://ieeexplore.ieee.org/document/18078)

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


## Features

- **Configurable Systolic Array**: Parameterized mesh dimensions, data types, and dataflows
- **Dual Dataflow Support**: Output-stationary (OS) and weight-stationary (WS) modes
- **Modern Python**: Clean, type-hinted Python 3.10+ codebase
- **Amaranth HDL**: Leverage Python's expressiveness for hardware design
- **Comprehensive Testing**: Unit tests with Amaranth's built-in simulator

## Installation

```bash
# Clone the repository
git clone https://github.com/stillwater/systars.git
cd systars

# Install in development mode
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- Amaranth HDL 0.5+

## Quick Start

```python
from systars.config import SystolicConfig, Dataflow
from systars.core.pe import PE
from amaranth.back import verilog

# Create a configuration
config = SystolicConfig(
    mesh_rows=16,
    mesh_cols=16,
    input_bits=8,
    acc_bits=32,
    dataflow=Dataflow.BOTH,
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
| Tile | 🔲 TODO | Combinational PE grid |
| Mesh | 🔲 TODO | Pipelined tile array |
| Scratchpad | 🔲 TODO | Multi-bank local memory |
| Accumulator | 🔲 TODO | Result memory with scale/activation |
| Controllers | 🔲 TODO | Load/Execute/Store command handling |
| DMA | 🔲 TODO | Memory transfer engines |

## Configuration

Key parameters in `SystolicConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mesh_rows` | 16 | Systolic array height |
| `mesh_cols` | 16 | Systolic array width |
| `input_bits` | 8 | Input element width |
| `acc_bits` | 32 | Accumulator width |
| `dataflow` | BOTH | WS, OS, or runtime selectable |
| `sp_capacity_kb` | 256 | Scratchpad size |
| `acc_capacity_kb` | 64 | Accumulator size |

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

## Project Structure

```
systars/
├── src/systars/
│   ├── config.py          # Configuration dataclass
│   ├── arithmetic/        # Arithmetic operations
│   ├── core/              # PE, Tile, Mesh
│   │   └── pe.py          # Processing Element
│   ├── memory/            # Scratchpad, Accumulator
│   ├── controller/        # Load/Execute/Store controllers
│   ├── dma/               # DMA engines
│   ├── util/              # Utilities (transposer, etc.)
│   └── loop/              # Loop unrollers
├── tests/
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── examples/              # Example configurations
├── scripts/               # Build/generation scripts
└── docs/                  # Documentation
```

## Roadmap

See [implementation plan](./doc/plan/implementation.md) for the detailed implementation plan.

### Phase 1: Foundation (Current)
- [x] Configuration system
- [x] Processing Element (PE)
- [ ] Tile
- [ ] Mesh

### Phase 2: Memory System
- [ ] Local address encoding
- [ ] Scratchpad banks
- [ ] Accumulator with scale/activation

### Phase 3: Controllers
- [ ] Reservation station
- [ ] Execute controller
- [ ] Load/Store controllers

### Phase 4: Loop Unrollers
- [ ] Matrix multiply loop
- [ ] Convolution loop

### Phase 5: Validation
- [ ] Header file generation
- [ ] Reference comparison

## License

BSD-3-Clause

## Acknowledgments

- [SYSTARS](https://ieeexplore.ieee.org/document/18078) - The original C implementation of the systolic array layout generator
- [Amaranth HDL](https://amaranth-lang.org/) - Python HDL framework
