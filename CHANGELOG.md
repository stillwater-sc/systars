# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

#### SystolicTop computes a GEMM end to end (2026-11-01)

Ported from branes-ai/embodied-ai-soc-generator#9, items 1–3 and 7 of #2. `SystolicTop` now
runs LOAD A and B, CONFIG, PRELOAD, COMPUTE and STORE C to a verified `A @ B`, output-stationary
and weight-stationary. The previous execute path could not:

- *B was never read*: both operands shared the scratchpad's one read port, and the address mux
  preferred A. The scratchpad has a second read port (in every bank), and A and B stream at one
  row per cycle
- *COMPUTE never completed*: it waited for an `out_last` pulse that had already passed, and
  dropped the outputs produced meanwhile
- *A, B and control were skewed inconsistently*: A was registered in every PE and at PEArray
  boundaries, B only at boundaries, and `valid` not at all across PEArray columns. Now every
  signal crossing a PEArray boundary passes through one register, and `SystolicArray` skews its
  inputs and deskews its outputs itself
- *B-stationary held no weight*: the PE multiplied the incoming partial sum. The PE now has a
  weight register, loaded by PRELOAD, and partial sums flow down
- *OS results could not be read out*: the PE ignored `in_d` in OS. Results now drain down each
  column through a shift chain, bottom row first

### Changed

- **Execute semantics**: `COMPUTE` overwrites its destination rows and `COMPUTE_ACCUMULATE` adds
  to them. OS computes `C[R x C] = A[R x K] @ B[K x C]` from Aᵀ and B rows; WS computes
  `C[M x C] = A[M x R] @ W` with `W` loaded by `PRELOAD` (which no longer reads the accumulator,
  and does nothing in OS). Details in `controller/execute.py`
- **PE ports**: `in_psum`/`out_psum` (WS partial sums) and `in_shift` (shift chain); the
  `propagate` double buffer, the rounding `shift`, `id` and the PE's pass-through registers are
  gone, and so is `PEWithShift` (unused)
- **Generated RTL**: the cocotb PEArray is 2x2, and `gen/systolic_array.v` is a 2x2 grid of 2x2
  PEArrays (4x4 PEs)
- **Unit-test simulations** (`tests/unit/_sim.py`) end when the testbench returns, with a cycle
  limit as a watchdog

#### Bugs found reviewing the kpugen port (2026-10-31)

These were found in code review of branes-ai/embodied-ai-soc-generator#1, which ports this
repository's systolic stack. Each RTL fix has a test that fails on the previous code.

- **Accumulator**: accumulate writes were lost when `accumulate` dropped after the request cycle
  (as `ExecuteController` drives it). Writes are now always pipelined, with a transparent
  read-modify-write read. Accumulate adds each element separately (no carries between packed
  elements), and activations apply per element (`memory/activation.py`)
- **Store/top**: activation was applied twice (store ReLU + accumulator read activation driven
  from the live command port) and to the whole packed row. It is now applied once, per element,
  from the latched command. Load/Store reject rows wider than `dma_buswidth` (were silently
  truncated)
- **SystolicTop**: per-controller completion ports; the summary `completed` drops simultaneous
  completions
- **ExecuteController**: CONFIG decodes the latched `rs1`, not the live port
- **DescriptorEngine**: transfers ended at half their length; multi-burst (> 256 beats) and
  zero-length transfers
- **StridedStreamReader / MatrixLoader**: could not elaborate (Python `if` on a `Signal`), packed
  elements at a 32-bit stride, and `MatrixLoader.busy` was never driven
- **SRAMReadScheduler**: missed the one-cycle `read_valid` pulse and hung
- **LocalAddr**: `make_sp_addr`/`make_acc_addr` used the bank value's bit length as the row shift;
  now instance methods using the configured bank bits
- **SystolicConfig** validates row/bank geometry; **Matmul/Conv2d** reject non-power-of-two dims;
  Conv2d no longer reports IDLE in `CALC_TILES`; `CONFIG_INPUT_DIMS` comment corrected
- **Examples**: int8 wraparound (03-06), scratchpad overlap in 01 for size > 16, `--step` crash
  in 02, workload-scaled cycle limit in conv 01, `run_all_gemm_demos.sh` runs from any directory
- **Scripts/CI**: OSS CAD Suite installs through a private temp dir, with archive verification and
  a staged replace (`bin/lib/oss-cad-suite.sh`); `$OSS_CAD_SUITE` honoured and on PATH for just
  recipes (after the venv); non-root `just` install; docs deploy concurrency and permissions;
  authenticated `setup-oss-cad-suite`; the Verilator pre-commit hook fails on errors; pytest
  discovery no longer collects cocotb tests
- **Tests**: PE tests now assert their results; new tests for accumulator row semantics,
  descriptor-engine transfers, the read scheduler, LocalAddr layout, and reader elaboration

### Known issues

- The execute path cannot yet compute a GEMM end to end (B operand never read in `SystolicTop`,
  COMPUTE result capture, Matmul/ExecuteController command encoding, ISA bank encoding, Conv2d
  addressing, array skew). This is tracked in a separate issue.

### Added

#### Repository Velocity Report Script (2025-12-23)

- **Velocity Report Generator** (`scripts/repo_velocity.py`): Measures development velocity across multiple repositories
  - Tracks source code files only: C/C++, Python, Go, Shell, CMake, Verilog/VHDL
  - Excludes large files (>100k lines) that are likely generated content
  - Reports: total LOC, commits, lines added/deleted, net change, contribution percentage
  - Multiple output formats: markdown (default), CSV, JSON
  - Configurable date range via `--since` and `--until` arguments
  - Supports both branes-ai and stillwater-sc organization repos

### Fixed

#### OSS-CAD-SUITE PATH in Virtual Environment (2025-12-23)

- **Venv Activation Script**: Fixed missing Yosys/Verilator in PATH after activating .venv
  - `.venv/bin/activate` now automatically adds `/opt/oss-cad-suite/bin` to PATH
  - RTL generation (`just gen`) and synthesis checks now work without manual environment sourcing

- **Setup Script Enhancement** (`bin/setup-dev-env.sh`): Auto-patches venv activation
  - New step patches `.venv/bin/activate` to include OSS-CAD-SUITE path
  - Idempotent: skips if already patched
  - Ensures `just ci` works immediately after `source .venv/bin/activate`

#### MasPar MP-2 SIMD Implementation (2025-12-21)

- **SIMD Package** (`src/systars/simd/`): New package for classic SIMD architectures
  - Distinct from SIMT (no divergence handling, pure lock-step execution)
  - MasPar MP-1/MP-2 as first implementation

- **MasPar Simulator** (`src/systars/simd/maspar/`): Complete MP-2 SIMD array processor
  - `MasParConfig`: Hardware configuration (64x64 array, 64 registers/PE, 64KB local memory)
  - `PE`: Processing Element with 32-bit ALU, registers, local memory
  - `PEArray`: 2D mesh with 8-neighbor XNET connectivity (toroidal wrap-around)
  - `ACU`: Array Control Unit for instruction fetch/decode/broadcast
  - `MasParSim`: Top-level simulator with GEMM and Conv2D support

- **Instruction Set** (`src/systars/simd/maspar/instruction.py`): 40+ opcodes
  - Integer ALU: IADD, ISUB, IMUL, IAND, IOR, IXOR, ISHL, ISHR, IMOV
  - Floating-point: FADD, FSUB, FMUL, FFMA
  - Memory: LD, ST, LDI (load immediate)
  - XNET: 8 directions (N, S, E, W, NE, NW, SE, SW)
  - Router: ROUTE_SEND, ROUTE_RECV
  - Control: SETMASK, CLRMASK (PE masking)
  - Reduction: REDUCE_SUM, REDUCE_MAX, REDUCE_MIN
  - Comparison: ICMP_EQ/NE/LT/LE/GT/GE

- **Matrix Multiplication (GEMM)**: Cannon's algorithm implementation
  - Pre-skewed data distribution for parallel execution
  - XNET shifts: A west, B north each iteration
  - Performance: 4x4 GEMM in 29 cycles (16 instructions)

- **2D Convolution (Conv2D)**: 3x3 kernel support
  - Leverages XNET 8-neighbor mesh for efficient gather
  - 36 instructions: 9 LDI + 8 XNET + 1 init + 18 MAC
  - Kernels: identity, box, edge, sharpen, sobel_x, sobel_y

- **Animation Examples** (`examples/simd/maspar/`):
  - `01_animated_maspar.py`: GEMM visualization with Cannon's algorithm
  - `02_animated_conv2d.py`: Conv2D visualization with kernel selection

- **Pre-defined Configurations**:
  - `DEFAULT_MASPAR_CONFIG`: 64x64 MP-2
  - `SMALL_MASPAR_CONFIG`: 4x4 for testing
  - `MEDIUM_MASPAR_CONFIG`: 16x16
  - `LARGE_MASPAR_CONFIG`: 128x128
  - `MP1_CONFIG`: 64x64 with 4-bit ALU (8 cycles for 32-bit ops)

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

#### SIMT Package Reorganization (2025-12-20)

- **Multi-Architecture Structure**: Reorganized `src/systars/simt/` to support multiple architectures
  - `simt/nv/`: NVIDIA Streaming Multiprocessor (current implementation)
  - `simt/nv_v1/`: Legacy NVIDIA SM with per-partition LSU bug
  - `simt/amd/`: AMD RDNA/CDNA placeholder (NotImplementedError)
  - `simt/maspar/`: MasPar MP-1/MP-2 placeholder (NotImplementedError)

- **Backward Compatibility Layer** (`src/systars/simt/__init__.py`):
  - Re-exports all NVIDIA symbols for existing code: `from systars.simt import SMSim`
  - Explicit architecture selection now possible: `from systars.simt.nv import SMSim`

- **Common Protocols** (`src/systars/simt/base.py`):
  - `ProcessorSim`: Protocol for top-level processor simulation
  - `MemorySubsystem`: Protocol for memory read/write operations
  - `SIMTConfigBase`: Abstract base for architecture configurations

- **Round-Robin Warp Distribution Fix**:
  - Changed from sequential fill (P0=8, P1=8, P2=2, P3=0 for 18 warps)
  - To balanced round-robin (P0=5, P1=5, P2=4, P3=4 for 18 warps)
  - Fixed in both `sm_controller.py` and `examples/simt/01_animated_simt.py`

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
