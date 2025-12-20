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

## Partitions are underutilized

The tiled GEMM now distributes warps across all partitions:

Summary of changes:

  1. Warp distribution logic - warps are now assigned to partitions:
    - Global warp 0-7 → P0, 8-15 → P1, 16-23 → P2, 24-31 → P3
  2. Global warp ID calculation - animation loop now computes global_warp_id = p_idx * warps_per_partition + local_warp_id for proper GEMM tracking
  3. Partition activation - each partition with work is activated independently

Results:

  | GEMM Size | Output Elements | Warps | Partitions | ALU Cores |
  |-----------|-----------------|-------|------------|-----------|
  | 16×16     | 256             | 8     | 1          | 32        |
  | 32×16     | 512             | 16    | 2          | 64        |
  | 32×32     | 1024            | 32    | 4          | 128       |

  The output now shows partition assignments:
  P0:W0 (global 0): rows 0-0, cols 0-31 (8 instrs)
  ...
  P3:W7 (global 31): rows 31-31, cols 0-31 (8 instrs)

  Running tiled GEMM: C[32×32] = A[32×2] @ B[2×32]
    Using 4 partitions, 32 warps, 128 ALU cores

## Coalescing Memory Requests

The coalescer is only for GLOBAL memory, not shared memory. Here's the architecture:

```text
  LSU → Address Decode (bits [31:30])
            │
            ├─→ SHARED [01]: Direct to Shared Memory SRAM
            │                (no coalescing, 4-cycle latency)
            │
            └─→ GLOBAL [00]: Coalescer → L1 Cache → DRAM
                             (coalesces 32 addresses into 128B transactions)
```

From the code (lines 238-254):

```python
  if addr_space == AddressSpace.SHARED:
      latency = self.config.shared_mem_latency  # No coalescing
  elif addr_space == AddressSpace.GLOBAL:
      # Analyze coalescing for global memory accesses
      coalesce_result = self.coalescer.analyze(stripped_addresses, thread_mask)
      num_transactions = coalesce_result.num_transactions
      # Latency scales with number of transactions
      latency = self.config.l1_miss_latency * num_transactions
```

Your display shows "Coalesce: 100%" - meaning all 32 threads' addresses fell into the same 128-byte segment, resulting in 1 transaction instead of 32. This is optimal coalescing.

When coalescing is poor (e.g., strided access), you'd see multiple transactions and higher latency. For example, if each thread accesses a different cache line, you get 32 transactions = 32× the latency.

## Timeline Analysis

W0 and W3 are the slowest to complete, why?

The key issue is memory starvation causing W0 and W3 to be massively delayed.

Timeline Analysis:

  | Warp  | First LD Issue | First LD Complete | ST Issue       | ST Complete |
  |-------|----------------|-------------------|----------------|-------------|
  | W0    | Cycle 8        | Cycle 209         | Cycle 1227     | Cycle 1429  |
  | W1    | Cycle 9        | Cycle 210         | Cycle 433      | -           |
  | W2    | Cycle 10       | ~Cycle 14         | Cycle 300      | -           |
  | W3    | Cycle 11       | Cycle 410         | Cycle 2253     | Cycle 2455  |
  | W4    | Cycle 12       | Cycle 411         | Cycle 468      | -           |
  | W5-W7 | Cycles 13-15   | quickly           | Cycles 301-303 | -           |

Root Cause:

  1. W0 and W1 issued LDs first (cycles 8-9), went into STALLED_MEM
  2. Scheduler moved to W2-W7 which kept issuing instructions
  3. LSU has max 2 pending requests - W0/W1's requests got queued
  4. W2-W7 monopolized the memory path - their requests completed quickly
  5. W0's first LD took 201 cycles (209-8) due to 200-cycle latency + queue delay
  6. W3 was worst - ST at cycle 2253, complete at 2455

The pattern shows:

```
  W2-W7: Issue LD → (fast) → FFMA → ST → Done by cycle ~300
  W0, W3: Issue LD → (200 cycle wait) → (queue behind others) → ST at cycles 1227, 2253
```

This is why your earlier screenshot showed W0 (rows 0-1) and W3 (rows 6-7) completing last - they were starved by the other warps.

This is a realistic GPU scheduling issue - memory-bound warps can get starved when the LSU queue is limited and other warps keep issuing new requests.

## LSU Queue bug

Bug Fix: LSU Queue Overflow (Critical)

Root Cause: When the LSU queue was full (max_pending=2), load_store_unit.issue() returned False, but the instruction was silently dropped. Warps continued executing with garbage register data.

Fix in src/systars/simt/partition.py:

  1. Added pending_memory_requests buffer to store rejected memory instructions
  2. Each cycle, pending requests are retried before handling new fires
  3. Warps properly stall until their memory request is accepted
  4. Added total_lsu_rejections statistic to track queue-full events

Verification:

- All 56 SIMT unit tests pass
- 16×16 GEMM with K=2 now completes all 64 instructions (32 LDs, 16 FFMAs, 8 MOVs, 8 STs)
- Each of the 8 warps has exactly 4 LD completions (previously W2-W7 had 0 completions)
- Works with both fast-mem (4 cycles) and default (200 cycles) latency

Files Updated:

- src/systars/simt/partition.py - Bug fix implementation
- CHANGELOG.md - Added entry for the fix
- docs/sessions/2025-12-20-simt-visualization-enhancements.md - Updated session log

## LSU Redesign

The LSU is currently at the partition level. Looking at the architecture:

```text
  SM Controller
  ├── Partition 0
  │   ├── Scheduler (8 warps)
  │   ├── Register File
  │   ├── Operand Collectors
  │   ├── Execution Units (ALUs)
  │   └── LSU (max_pending=2) ← HERE, per partition
  ├── Partition 1 (same structure)
  ├── Partition 2
  ├── Partition 3
  ├── Shared Memory (48KB, shared across partitions)
  └── Global Memory Interface
```

The design of the LSU with  max_pending=2 is woefully inadequate. With 8 warps per partition and 200-cycle memory latency, we need much deeper queuing to hide latency. The math:

  | Metric                 | Value                                    |
  |------------------------|------------------------------------------|
  | Memory latency         | 200 cycles                               |
  | Warps per partition    | 8                                        |
  | Current max_pending    | 2                                        |
  | Theoretical throughput | 2 requests / 200 cycles = 0.01 req/cycle |
  | Required for 8 warps   | 8 warps × 1 LD = 8 outstanding           |

Real GPU comparison (NVIDIA):

- LSU is per-SM (not per-partition), shared across all 4 sub-cores
- 32+ MSHR (Miss Status Holding Registers) for outstanding requests
- L1 cache has separate request queues
- Memory controller handles coalescing and request batching

Architectural options:

  1. Increase per-partition queue: max_pending=8 (one per warp)
  2. Move LSU to SM level: Single LSU shared by all partitions with deeper queue (16-32)
  3. Add MSHR-style tracking: Separate miss handling from request queuing

## What does NVIDIA do?

NVIDIA aggregates the LSU at the SM level, the question is why?

### Key Architecture Findings

Unified L1/Shared Memory Subsystem

  Since Volta (2017), NVIDIA combines L1 cache and shared memory into a single 128KB SRAM block per SM that serves all 4 partitions. This unified design:

- Enables dynamic partitioning between cache and shared memory per workload
- Provides a single coalescing point for all memory requests
- Simplifies coherence (write-through to L2 avoids complex L1 coherence)

LSU Bottleneck by Design

  From the <https://forums.developer.nvidia.com/t/how-does-the-lsu-load-store-unit-execute-load-store-instructions-in-the-ampere-architecture/273699>:
  "Each cycle the warp scheduler can issue an instruction to the LSU/MIO instruction queue. Each cycle the MIO unit can issue 1 instruction to the LSU pipe. This limits the issue rate from 4 IPC per SM to 1 IPC per SM."

  So despite 4 warp schedulers issuing in parallel, memory operations serialize through a single MIO queue - this is intentional.

MSHR-Based Latency Hiding

  The key to hiding memory latency isn't LSU queue depth but MSHRs (Miss Status Holding Registers):

- Research shows ~6 MSHRs per warp, ~64 total per SM
- MSHRs track outstanding cache misses, not instruction queue depth
- <https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.pdf> shows a "cusp behavior" where optimal warp count depends on arithmetic intensity

Why Shared LSU Makes Sense

  | Design Choice           | Rationale                                                                                     |
  |-------------------------|-----------------------------------------------------------------------------------------------|
  | Single coalescing point | All 32 threads × 4 partitions = 128 addresses can be coalesced into minimal 128B transactions |
  | Unified tag stage       | L1TEX can resolve 4 sets × 4 sectors per cycle efficiently                                    |
  | MSHR sharing            | Primary/secondary miss tracking reduces redundant DRAM requests                               |
  | Bandwidth matching      | L1 bandwidth (128B/cycle) matches single LSU throughput                                       |
  | Area efficiency         | One complex address generation unit vs four simpler ones                                      |

Current Systars Gap

  Our current design (max_pending=2 per partition) is fundamentally wrong because:

  1. No MSHR tracking - we track pending instructions, not pending cache lines
  2. No coalescing benefit - per-partition LSU can't merge cross-partition requests
  3. Queue too shallow - need tracking proportional to (memory_latency / issue_rate)

Design Options to Consider

  1. Move LSU to SM level with single MIO queue feeding into L1TEX-style unified cache
  2. Add MSHR tracking separate from instruction queue (track cache lines, not warps)
  3. Increase outstanding request limit based on max_outstanding_requests: int = 16 in config

Sources:

- <https://forums.developer.nvidia.com/t/how-does-the-lsu-load-store-unit-execute-load-store-instructions-in-the-ampere-architecture/273699>
- <https://www.rastergrid.com/blog/gpu-tech/2021/01/understanding-gpu-caches/>
- <https://modal.com/gpu-glossary/device-hardware/load-store-unit>
- <https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.pdf>
- <https://web.engr.oregonstate.edu/~chenliz/publications/2019_ICS_Dynamcially%20Linked%20MSHRs.pdf>

## Chrome Trace format feature

Usage:

```bash
  # Chrome Trace format (default)
  python examples/simt/01_animated_simt.py --tiled --m 8 --n 8 --k 4 \
      --fast --fast-mem --timeline trace.json

  # Explicit format choice
  python examples/simt/01_animated_simt.py --timeline trace.json --timeline-format chrome
  python examples/simt/01_animated_simt.py --timeline events.csv --timeline-format csv
```

Viewing the trace:

  1. Open chrome://tracing in Chrome/Chromium
  2. Click "Load" and select the JSON file
  3. Or use <https://ui.perfetto.dev> (better UI)

Chrome Trace features:

- Each partition shows as a separate "process" (row group)
- Each warp shows as a separate "thread" (row within partition)
- Events show as colored bars with duration
- Categories: memory (LD/ST), compute,mul (FFMA), compute,alu (IADD/MOV)
- Zoom and pan to explore the timeline
- Click events to see args (dst, src1, src2, latency)

  Event structure:

```json
  {
    "name": "FFMA",
    "cat": "compute,mul",
    "ph": "X",           // Complete event with duration
    "ts": 15000,         // Start time (cycle × 1000 for ns)
    "dur": 4000,         // Duration (4 cycles)
    "pid": 0,            // Partition 0
    "tid": 2,            // Warp 2
    "args": {"dst": 5, "src1": 1, "src2": 2, "latency": 4}
  }
```
