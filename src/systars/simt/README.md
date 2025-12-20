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
