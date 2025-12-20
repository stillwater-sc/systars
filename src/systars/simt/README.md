# Single Instruction Multiple Thread

```bash
  # Basic animation with 4 warps, 200ms delay between frames
  python examples/simt/01_animated_simt.py --warps 4 --delay 200

  # Fast mode (no delay) with 8 warps and more instructions
  python examples/simt/01_animated_simt.py --warps 8 --instructions 32 --fast

  # GEMM workload animation (matrix multiply inner loop)
  python examples/simt/01_animated_simt.py --gemm --k 16 --warps 4 --delay 150

  # Slow detailed view for studying pipeline behavior
  python examples/simt/01_animated_simt.py --warps 2 --instructions 8 --delay 500

  # Energy comparison (non-animated, shows efficiency breakdown)
  python examples/simt/02_energy_comparison.py
```

The animation shows:

- Warp schedulers: R=Ready, E=Executing, W=RAW-Stall, B=Bank-Stall, D=Done
- Register file banks: ██=Read, ▓▓=Write, XX=Conflict, ░░=Idle
- Operand collectors: ■=collected, □=pending, ·=empty
- Energy breakdown: Real-time efficiency tracking (~5% for SIMT vs 75% systolic)

## Summary of Changes

  1. Fixed Operand Collectors (now show activity)

- Added READING state for proper 2-phase collection with 1-cycle register file latency
- Collectors now transition: PENDING → READING (◐) → READY (■)
- Status shows 0+2/2 format (ready + reading / total)

  2. Enhanced Execution Unit (8 ALUs per partition)

- Each partition now has 8 ALUs (matching cores_per_partition config)
- Each ALU has a 4-stage pipeline with proper stage tracking
- Round-robin ALU selection for instruction distribution

  3. Pipeline Visualization

- Shows all 8 ALUs per partition (A0-A7) with busy/idle state
- Displays ALU utilization percentage
- Shows pipeline stage occupancy with █ for active, · for idle

  4. Unit Tests (37 tests, all passing)

- TestSIMTConfig: Configuration validation
- TestRegisterFileBank: RTL register file bank
- TestRegisterFileSim: Behavioral register file with bank conflicts
- TestWarpSchedulerSim: Round-robin scheduling
- TestOperandCollectorSim: Two-phase operand collection
- TestALUPipeline: Individual ALU pipeline behavior
- TestExecutionUnitSim: 8-ALU cluster operation
- TestPartitionSim: Full partition integration
- TestGEMMProgram: GEMM execution
- TestExecutionUnitRTL: RTL execution unit

  5. Functional Correctness Verified

- SIMT model executes programs correctly
- 4 warps × 8 instructions = 32 instructions completed
- IPC of 0.89 with realistic stall modeling
- Energy tracking works correctly

## Animation Commands

```bash
  # See operand collectors and ALU pipelines in action
  python examples/simt/01_animated_simt.py --warps 4 --delay 200

  # Fast mode with GEMM workload
  python examples/simt/01_animated_simt.py --gemm --k 8 --fast
```

## Summary of SIMT Memory architecture

## Phase 1: Core Memory Path

- config.py: Extended with memory parameters (unified cache, shared memory banks, latencies, address space config)
- shared_memory.py: New file - Banked shared memory with conflict detection
- load_store_unit.py: New file - LSU with address space decoding and routing
- partition.py: Integrated LSU, routes LD/ST to memory instead of ALU

## Phase 2: Memory Coalescing

- global_memory.py: New file - Dict-based DRAM simulation with matrix operations
- memory_coalescer.py: New file - Analyzes 32 thread addresses, groups into 128B transactions

## Phase 3: Barrier Synchronization

- barrier.py: New file - __syncthreads() implementation with warp arrival tracking

## Phase 4: Functional GEMM Test

- test_simt_memory.py: 19 new tests covering:
  - Shared memory read/write, bank conflict detection
  - Global memory matrix operations
  - Memory coalescing analysis
  - SM memory sharing between partitions
  - Functional GEMM with NumPy verification

## Phase 5: Energy Model Extension

- energy_model.py: Extended to include memory costs (shared, L1, DRAM)
- Added energy_from_sm_statistics() for post-simulation analysis

## Test Results

- 327 tests pass (308 original + 19 new memory tests)
- All linting checks pass

## New Architecture Components

```text
  SM Controller
  ├── Global Memory (DRAM simulation)
  ├── Shared Memory (SM-wide, 32 banks)
  ├── Barrier Unit
  └── Partitions (4)
      ├── Warp Scheduler (STALLED_MEM support)
      ├── Register File
      ├── Operand Collector
      ├── Execution Unit
      └── Load/Store Unit → Memory Coalescer → Shared/Global Memory
```

## Changes Made

1. Timeline logging for issue/complete events (examples/simt/01_animated_simt.py)

- Added TimelineLogger class that writes CSV format logs with columns: cycle,event,partition,warp,opcode,dst,src1,src2,latency,info
- Added --timeline FILE command line option to enable logging
- Timeline logs capture: ISSUE events when instructions are dispatched, COMPLETE events when ALU/LD/ST finish

2. Fixed matrix to distinguish computed vs stored elements

- Updated GEMMTracker with record_store_issued() and record_store_complete() methods
- Matrix now shows 4 distinct states:
  - W# (dim) = not started
  - ░░ / ▓░ = partial progress (some FFMAs done)
  - ▓▓ (yellow) = computed but ST pending
  - ██ (green) = truly complete (stored to memory)
- Header shows status: [computing], [ST pending: N], or [COMPLETE]
- Summary line shows both "Computed: X/Y" and "Stored: X/Y"

Example usage:

```bash
  # Run with timeline logging and see store completion tracking
  python examples/simt/01_animated_simt.py --tiled --m 8 --n 4 --k 2 --fast --fast-mem --timeline timeline.csv

  # Analyze timeline to find bubbles
  cat timeline.csv
```
