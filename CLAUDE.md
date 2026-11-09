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
just gen                    # Generate every target in systars.targets into gen/
just gen-pe                 # Generate PE Verilog
just gen-pe-array           # Generate PEArray Verilog (1x1, 2x2)
just gen-systolic-array     # Generate SystolicArray Verilog (2x2 of 2x2, 4x4)
python3 -m systars.targets list   # The targets: name, top module, description
just wavefronts             # Regenerate the docs' wavefront animations (docs/wavefront/)
```

### Synthesis and Waveforms

```bash
just synth-check            # Run Yosys synthesis check on generated RTL
just synth <target>         # Map every target onto generic|ice40|ecp5|polarfire cells, print stats
just synth-report           # Markdown resource report for every target (resource_report.md)
just waves                  # Open waveform viewer (gtkwave)
```

### Documentation

```bash
just docs                   # Build the Starlight site (docs-site/; Node >= 22.23.4)
just docs-serve             # Serve it locally with live reload
```

The GitHub Pages site uses Astro Starlight, the same setup as stillwater-sc/universal and
stillwater-sc/mtl5 (never MkDocs). Write pages in `docs/` as plain Markdown with an H1 title;
`docs-site/sync-content.mjs` generates `docs-site/src/content/docs/` (git-ignored, never edit
it). A page at the top of `docs/` needs a `ROOT_PAGES` entry in `sync-content.mjs`; pages in
subdirectories are placed by directory. Broken relative links and anchors fail the build.
Details: `docs/development-workflow.md`, "Documentation".

## Architecture

### Core Components (src/systars/)

The architecture follows a hierarchical design:

```
PE (Processing Element) → PEArray (PE grid) → SystolicArray (PEArray grid) → Top-level accelerator
```

- **PE (`core/pe.py`)**: The MAC unit, with an input register on every link, so values move one PE per clock: A east (with `valid`/`first`/`last` markers), B and partial sums south, OS results north. Two modes, selected at run time by `in_control_dataflow`:
  - Output-stationary (0): `c += a * b`; `first` overwrites, so a reduction may span commands. On `last` the PE launches `c + a * b` north into the `c_up` register of the PE above it
  - Stationary (1; WS and IS): a stationary register `w`, loaded through a registered chain down each column (`in_shift`, `in_w` -> `out_w`); the streamed operand moves east and `out_psum = in_psum + a * w` moves south

- **PEArray (`core/pe_array.py`)**: A structural block of PEs (tile_rows × tile_cols). Every PE link is registered, so PEArray boundaries make no difference to the timing.

- **SystolicArray (`core/systolic_array.py`)**: Grid of PEArrays (grid_rows × grid_cols); it scales from 8x8 up to 64x64 PEs. The array has no skew or deskew registers: the L1 stream buffers stagger the streams, and the L1 result buffers turn result columns back into rows. Per-edge result clocks: `os_result_clock`, `stationary_result_clock`.

- **L1 (`memory/stream_buffer.py`)**: `StreamBuffers` (one block per array row or column, one credit each; a buffer takes the next block in the clock its last element leaves) and `ResultBuffers` (one FIFO per column).

- **ExecuteController (`controller/execute.py`)**: The Streamer. It reads L2 (the scratchpad's two read ports) into L1 on schedule and never stalls: with `K >= R, C` tiles stream back to back, and a reduction's end waits for the OS north drain (`2R - 1` clocks between reductions). It reports each command's L2 release (`completed`, once its last block is in L1) and its result (`result`) separately. The command semantics (CONFIG, PRELOAD, COMPUTE, COMPUTE_ACCUMULATE; `cmd_chain`; OS, WS and IS operand layouts) are in its module docstring.

- **Config (`config.py`)**: Central `SystolicConfig` dataclass with all hardware parameters. Pre-defined configs: `DEFAULT_CONFIG`, `LEAN_CONFIG`, `CHIP_CONFIG`, `LARGE_CONFIG`

### Implemented Components

- PE, PEArray, SystolicArray (core compute hierarchy)
- Scratchpad (two read ports) and Accumulator memory banks
- Load/Execute/Store controllers, DMA engines (StreamReader/StreamWriter), and `SystolicTop`,
  which computes a verified GEMM end to end (`tests/unit/test_gemm_systolic.py`)
- `SystolicTop` is a systolic KPU compute tile following kpu-sim's CSP model: AXI = L3,
  scratchpad = L2, the stream and result buffers at the array's edges = L1; LOAD/STORE are the BlockMover (L3 <-> L2) and
  the execute controller is the Streamer (L2 -> L1). A LOAD or STORE moves a whole tile from
  or to any byte address (`cmd_stride`, `cmd_row_bytes`; `tests/unit/test_block_mover.py`),
  with a corner turn when its opcode is `DmaOpcode.TRANSPOSE`. No burst crosses a 4 KB
  boundary (`tests/unit/_axi.py` enforces it). A LOAD to an accumulator
  address (`LocalAddr` bit 31; bit 30 adds) writes a result tile's starting values, in the
  clocks the Streamer leaves the write port free; `CLEAR` (0x05) writes zero rows with no L3
  read, `BROADCAST` (0x06) one L3 row to every row, `BROADCAST_TRANSPOSE` (0x07) with a
  corner turn.
  Use KPU/CSP terms (credits, push, tile-ready tokens), never cache terminology
- Credits and tile-ready tokens (`controller/sync.py`): `SystolicTop` queues each process's
  commands and starts one only when its whole-tile slot credits and tokens are there
  (`cmd_sync`; `tests/unit/test_credits.py`). A LOAD into the accumulator is the
  accumulator pool's second producer (`SlotPool(producers=2)`, the `fill_*` ports)
- Wavefront tracing and animations (`wavefront.py`): records the `SystolicArray` RTL on the
  Streamer's schedule, checks every PE in every clock against the predicted wavefront, and
  draws value / index / front views as animated SVGs. The docs embed a committed
  progression (`docs/wavefront/`, 2x2 -> 4x4 -> 16x16, `wavefront.STAGES`): after changing
  the fabric, the Streamer's schedule or the renderer, run `just wavefronts` and commit the
  SVGs (`tests/unit/test_wavefront.py` fails when they are stale)
- Matmul and Conv2d ISA instructions (`isa/`): tile an operation larger than the array and
  push `SystolicTop` commands ahead, double-buffered (`isa/command.py`); Matmul runs OS, WS
  or IS (`CONFIG_DATAFLOW`) and Conv2d runs OS. A bias, or accumulating onto the existing
  output, starts each output tile from an accumulator LOAD (all three dataflows); Conv2d's padding is
  a `CLEAR`; see
  `docs/isa-instructions.md`

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
