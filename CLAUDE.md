# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Systars is a Python-based systolic array RTL generator using Amaranth HDL. It generates synthesizable Verilog for configurable systolic array accelerators targeting linear algebra workloads. This is a modern Python reimplementation inspired by the original SYSTARS work at Delft University.

## Commands

### Setup

```bash
# Full automated setup (downloads OSS-CAD-SUITE, creates venv, installs deps)
./bin/setup-dev-env.sh

# Or if OSS-CAD-SUITE is already installed:
just setup

# Activate in a new shell:
source .venv/bin/activate
```

OSS-CAD-SUITE provides EDA tools (Yosys, Verilator). Set `OSS_CAD_SUITE` env var if installed elsewhere (default: `/opt/oss-cad-suite`).

### Testing

```bash
just test-unit              # Run unit tests (pytest)
just test-cocotb            # Run cocotb simulation tests with Verilator
just test-file <path>       # Run a specific test file
just test-k <pattern>       # Run tests matching pattern
just test-cov               # Run tests with coverage report
```

### Linting and Formatting

```bash
just lint                   # Run all linters (Python + RTL)
just lint-python            # Lint Python with ruff
just lint-rtl               # Lint generated Verilog with Verilator
just format                 # Format Python code with ruff
just typecheck              # Run mypy type checking
```

### RTL Generation

```bash
just gen                    # Generate all RTL
just gen-pe                 # Generate PE Verilog (scripts/gen_pe.py)
just gen-pe-array           # Generate PEArray Verilog (scripts/gen_pe_array.py)
just gen-systolic-array     # Generate SystolicArray Verilog (scripts/gen_systolic_array.py)
```

### Synthesis and Waveforms

```bash
just synth-check            # Run Yosys synthesis check on generated RTL
just waves                  # Open waveform viewer (gtkwave)
```

## Architecture

### Core Components (src/systars/)

The architecture follows a hierarchical design:

```
PE (Processing Element) → PEArray (PE grid) → SystolicArray (PEArray grid) → Top-level accelerator
```

- **PE (`core/pe.py`)**: The MAC unit. Two dataflows, selected at run time by `in_control_dataflow`:
  - Output-stationary (0): `c += a * b` while `in_valid`; A and B pass through combinationally. Results leave by shifting down the column (`in_shift`, `in_d` -> `out_c`), bottom row first
  - Weight-stationary (1): a weight register `w`, loaded through the same shift chain; A flows right and partial sums flow down: `out_psum = in_psum + a * w`
  - No pass-through registers: pipeline registers sit only at PEArray boundaries

- **PEArray (`core/pe_array.py`)**: Combinational grid of PEs (tile_rows × tile_cols). Data flows directly between adjacent PEs without pipeline registers.

- **SystolicArray (`core/systolic_array.py`)**: Pipelined grid of PEArrays (grid_rows × grid_cols). Every signal crossing a PEArray boundary passes through exactly one register. The array skews its own inputs and deskews its outputs, so callers present unskewed A/B vectors; `latency = (grid_rows - 1) + (grid_cols - 1)`.

- **ExecuteController (`controller/execute.py`)**: Streams A and B from the scratchpad's two read ports into the array and writes result rows to the accumulator. The command semantics (CONFIG, PRELOAD, COMPUTE, COMPUTE_ACCUMULATE; OS and WS operand layouts) are in its module docstring.

- **Config (`config.py`)**: Central `SystolicConfig` dataclass with all hardware parameters. Pre-defined configs: `DEFAULT_CONFIG`, `LEAN_CONFIG`, `CHIP_CONFIG`, `LARGE_CONFIG`

### Implemented Components

- PE, PEArray, SystolicArray (core compute hierarchy)
- Scratchpad (two read ports) and Accumulator memory banks
- Load/Execute/Store controllers, DMA engines (StreamReader/StreamWriter), and `SystolicTop`,
  which computes a verified GEMM end to end (`tests/unit/test_gemm_systolic.py`)
- Matmul and Conv2d ISA instructions (`isa/`); their command encoding does not yet match
  the execute controller (#2)

### Planned Components (see docs/plan/implementation.md)

- Loop unrollers (LoopMatmul, LoopConv)

### Test Structure

- **Unit tests** (`tests/unit/`): Amaranth simulator-based tests using `amaranth.sim`
- **Cocotb tests** (`verif/cocotb/tests/`): RTL simulation tests with Verilator

### Amaranth HDL Patterns

This codebase uses Amaranth 0.5+ with `lib.wiring.Component`:

```python
from amaranth.lib.wiring import Component, In, Out

class PE(Component):
    def __init__(self, config: SystolicConfig):
        super().__init__({
            "in_a": In(signed(config.input_bits)),
            "out_c": Out(signed(config.acc_bits)),
        })

    def elaborate(self, _platform):
        m = Module()
        # Combinational: m.d.comb += ...
        # Registered: m.d.sync += ...
        return m
```

Verilog generation:

```python
from amaranth.back import verilog
output = verilog.convert(pe, name="PE")
```

## Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_rows/cols` | 16 | SystolicArray dimensions (number of PEArrays) |
| `tile_rows/cols` | 1 | PEArray dimensions (PEs per tile) |
| `input_bits` | 8 | Input element width |
| `acc_bits` | 32 | Accumulator width |
| `dataflow` | OUTPUT_STATIONARY \| B_STATIONARY | Supported dataflow modes (Flag, can combine) |
| `sp_capacity_kb` | 256 | Scratchpad size |
| `acc_capacity_kb` | 64 | Accumulator size |
