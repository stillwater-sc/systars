# Systars

A Python-based systolic array RTL generator using [Amaranth HDL](https://amaranth-lang.org/).

Systars generates synthesizable Verilog for configurable systolic array accelerators, targeting linear algebra workloads in signal processing, sensor fusion, and real-time control.

## Quick Start

```bash
# Install
git clone https://github.com/stillwater-sc/systars.git
cd systars
just setup
source .venv/bin/activate

# Run tests
just test
```

## Features

- **Configurable Systolic Array**: Parameterized grid dimensions, data types, and dataflows
- **Dual Dataflow Support**: Output-stationary (OS) and weight-stationary (WS) modes
- **Modern Python**: Clean, type-hinted Python 3.10+ codebase
- **Amaranth HDL**: Leverage Python's expressiveness for hardware design

See the [README](https://github.com/stillwater-sc/systars) for full documentation.
