# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### SM-Level LSU with MSHR Tracking (2025-12-20)

- **SM-Level Load/Store Unit** (`src/systars/simt/sm_lsu.py`): Restructured from per-partition LSUs to single SM-level LSU
  - Single MIO queue (16 entries) shared by all 4 partitions
  - 64 MSHRs tracking cache lines at 128B granularity
  - Primary/secondary miss optimization: multiple warps share same MSHR for same cache line
  - 1 IPC processing bandwidth to LSU pipe
  - Shared memory bypass (no MSHR tracking needed)
  - Statistics: `total_primary_misses`, `total_secondary_misses`, `total_mio_queue_full_stalls`

- **MSHR Data Structures**:
  - `MSHREntry`: Tracks cache line address, state, waiters, cycles remaining
  - `MSHRWaiter`: Partition ID, warp ID, thread mask, destination register, thread offsets
  - `MIOQueueEntry`: Queued memory request with partition/warp info, addresses, data

- **Config Parameters** (`src/systars/simt/config.py`):
  - `num_mshrs: int = 64` - Number of Miss Status Holding Registers per SM
  - `mio_queue_depth: int = 16` - Depth of MIO queue
  - `lsu_issue_bandwidth: int = 1` - Requests processed per cycle

- **Architecture Benefits**:
  - 64 MSHRs vs previous 8 outstanding requests (4 partitions × 2 pending)
  - Secondary miss optimization reduces redundant memory traffic
  - Cross-partition visibility for better load balancing
  - Matches NVIDIA's proven SM architecture

#### SIMT Visualization Enhancements (2025-12-20)

- **Timeline Logging** (`examples/simt/01_animated_simt.py`): Pipeline event tracing
  - `TimelineLogger` class for CSV-format event capture
  - Logs ISSUE, COMPLETE, and STALL events with cycle-accurate timing
  - `--timeline FILE` command line option to enable logging
  - Summary statistics printed at end of simulation

- **Store Completion Tracking**: Distinguish computed vs stored elements
  - `GEMMTracker` now tracks pending stores per warp
  - Matrix header shows status: `[computing]`, `[ST pending: N]`, or `[COMPLETE]`
  - Summary shows both "Computed: X/Y" and "Stored: X/Y" counts

- **Updated Matrix Visualization**: Clear visual state progression
  - `W#` (dim) = not started
  - `░░` / `▓░` = partial progress (some FFMAs done)
  - `▓▓` (yellow) = computed but ST pending
  - `██` (green) = truly complete (stored to memory)

### Fixed

#### Multi-Partition Warp Distribution (2025-12-20)

- **Warp Distribution Bug**: Fixed tiled GEMM loading all warps into partition 0 only
  - Now distributes warps across all 4 partitions (P0-P3)
  - 32×32 GEMM now uses all 128 ALU cores instead of just 32
  - Added global warp ID calculation for proper GEMM tracking across partitions

#### LSU Queue Overflow Bug (2025-12-20)

- **Memory Instruction Drops**: Fixed critical bug where LD/ST instructions were silently dropped
  - When LSU queue was full (max_pending=2), `issue()` returned False and instruction was lost
  - Warps continued executing with garbage register data, causing incorrect results
  - Fix: Added `pending_memory_requests` buffer to retry rejected instructions on subsequent cycles
  - Warps now properly stall until their memory request is accepted by LSU
  - Added `total_lsu_rejections` statistic to track queue-full events

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
