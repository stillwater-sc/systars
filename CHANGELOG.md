# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### ISA-Level Matmul Instruction (2025-12-18)

- **Matmul ISA Instruction** (`src/systars/isa/matmul.py`): High-level tiled matrix multiply
  - Configuration interface: dimensions (M, N, K), addresses, strides, options
  - Automatic dataflow selection: Output-Stationary, A-Stationary, B-Stationary
  - Tiled computation with edge tile handling for arbitrary matrix sizes
  - Internal command generation: LOAD_A, LOAD_B, LOAD_D, EXEC_CONFIG, EXEC_PRELOAD, EXEC_COMPUTE, STORE_C
  - Backpressure handling via valid/ready handshake on command interface
  - Double buffering with bank toggling for latency hiding

- **ISA Module** (`src/systars/isa/__init__.py`): New ISA instructions package

- **Matmul Example** (`examples/gemm/07_isa_matmul.py`): Interactive demonstration
  - Shows configuration, tiling, command generation, and dataflow selection
  - Supports configurable M, N, K dimensions and array size
  - Validates command counts against expected values

- **Matmul Unit Tests** (`tests/unit/test_matmul.py`): 13 comprehensive tests
  - Configuration interface tests
  - Address calculation for tiled operations
  - Dataflow selection heuristics
  - Command emission verification
  - Backpressure handling
  - Double buffering bank toggling
  - Multi-tile execution
  - Bias loading (LOAD_D)
  - Error handling

- **Implementation Plan** (`docs/plan/isa-instructions.md`): Design documentation for Matmul and Conv2d

### Changed

#### CI Improvements (2025-12-18)

- **Unit Tests CI**: Added OSS CAD Suite (Yosys) to unit tests job for full Verilog generation test coverage
- **Dependencies**: Added `numpy>=1.24` to dev dependencies for GEMM demo/e2e tests

### Fixed

#### Verilog Generation Tests (2025-12-18)

- Added skip-if-no-Yosys logic to 3 tests in `test_gemm_demo.py` and `test_gemm_e2e.py`
- All 23 Verilog generation tests now properly skip when Yosys unavailable (safety net)

#### Skew Buffer Timing Visualization (2025-12-17)

- **Interactive Step Mode**: `--step` flag for manual single-stepping through animation
- **Dynamic Sizing**: `--k` now defaults to `--size` for intuitive square matrix behavior
- **Timeline Logging**: Cycle-by-cycle event log with throughput metrics after validation

### Fixed

#### Skew Buffer Display Fixes (2025-12-17)

- **SRAM Timing Display**: Fixed k-value calculation showing k=-1 when data became valid
- **Column Alignment**: Fixed Skew A (4-char) and Skew B (5-char) column width consistency
- **Dot-Value Alignment**: Changed dot position to align with value ones digit
- **Dynamic Left Margin**: Margin now scales with array size (`7 + max_depth * 4`)
- **FIFO Register Mapping**: Fixed slot assignment so data enters at deepest register, not R0
  - Formula: `R[d]` displays `fifo[d - depth + fifo_len]`
- **Display Timing**: All state now captured and displayed consistently after `step()`

#### Examples and Wavefront Visualization (2025-12-16)

- **Examples Framework** (`examples/`): User-facing demonstration applications
  - `examples/common/dram_model.py`: Simulated DRAM with matrix load/store operations
  - `examples/common/tensor_utils.py`: Matrix packing, unpacking, and tiling utilities
  - `examples/gemm/01_simple_matmul.py`: Complete matrix multiply demo with command generation
  - `examples/gemm/02_animated_wavefront.py`: Terminal animation of systolic wavefront
  - `examples/gemm/03_wavefront_gif.py`: GIF generator for shareable animations

- **Wavefront Documentation** (`docs/wavefront-animation.md`): Comprehensive guide
  - Theory of wavefront data flow and skewing patterns
  - Cycle-by-cycle breakdown of systolic array operation
  - Visualization tool usage and color coding guide
  - GIF generation for Slack/documentation sharing

- **End-to-End Integration Tests** (`tests/unit/test_gemm_e2e.py`)
  - `SimulatedAXIDRAM` class for testing with inline AXI handling
  - Tests for load, store, execute, and full GEMM sequence
  - Verification against NumPy reference computations

- **Top-Level Integration** (`src/systars/top.py`): SystolicTop module
  - Wires all accelerator components together
  - Command dispatch to load/store/execute controllers
  - AXI interface connection to StreamReader/StreamWriter

#### Phase 2: Memory System (2025-12-15)

- **LocalAddr** (`src/systars/memory/local_addr.py`): Unified address encoding utilities
  - Static methods for bit field extraction: `is_acc()`, `accumulate()`, `read_full_row()`, `is_garbage()`
  - Instance methods for bank/row decoding: `sp_bank()`, `sp_row()`, `acc_bank()`, `acc_row()`
  - Address construction helpers: `make_sp_addr()`, `make_acc_addr()`, `make_garbage_addr()`

- **ScratchpadBank** (`src/systars/memory/scratchpad.py`): Single bank local memory
  - Configurable read latency pipeline
  - Byte-level write masking with granularity=8
  - Read/write ports with valid signaling

- **Scratchpad** (`src/systars/memory/scratchpad.py`): Multi-bank scratchpad controller
  - Address-based bank selection routing
  - Parallel bank instantiation
  - Pipelined bank selection to match read latency

- **AccumulatorBank** (`src/systars/memory/accumulator.py`): Single bank accumulator memory
  - Activation functions: NONE, RELU, RELU6
  - Accumulate mode (read-modify-write) for partial sum accumulation
  - Signed data storage with configurable width

- **Accumulator** (`src/systars/memory/accumulator.py`): Multi-bank accumulator controller
  - Bank selection via address decoding
  - Activation function passthrough to banks

- **Unit Tests**: 34 new tests for memory subsystem
  - `tests/unit/test_local_addr.py`: 11 tests for address encoding
  - `tests/unit/test_scratchpad.py`: 12 tests for scratchpad memory
  - `tests/unit/test_accumulator.py`: 11 tests for accumulator memory

#### Phase 1: Core Systolic Array (Previous)

- **ProcessingElement (PE)**: MAC unit with input/weight propagation
- **PEArray**: Combinational grid of PEs with systolic data flow
- **SystolicArray**: Top-level pipelined grid of PEArrays
- **SystolicConfig**: Centralized configuration dataclass
