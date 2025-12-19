# SIMT Streaming Multiprocessor Architecture

This document describes the NVIDIA-style Streaming Multiprocessor (SM) implementation
used as a baseline for demonstrating the energy efficiency advantages of systolic arrays
and stencil machines.

## Overview

The SIMT (Single Instruction Multiple Thread) architecture executes the same instruction
across multiple threads (a warp) simultaneously. While highly flexible, this approach
incurs significant instruction overhead compared to dataflow architectures.

## Architecture Specification

```
Streaming Multiprocessor (SM)
├── 4 Sub-Cores (Partitions/Processing Blocks)
├── 32 CUDA Cores total (8 per partition)
├── 4 Warp Schedulers (1 per partition)
├── 64K Registers total (256KB)
│   └── 16K Registers per partition (64KB per partition)
│   └── 16 Banks per partition (4KB per bank)
├── Operand Collectors (consolidator registers)
└── Shared Memory / L1 Cache (configurable)
```

## Block Diagram

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │              STREAMING MULTIPROCESSOR (SM)                   │
                    │                                                              │
    ┌───────────────┼──────────────────────────────────────────────────────────────┤
    │ INSTRUCTION   │  ┌─────────────────────────────────────────────────────────┐ │
    │ CACHE         │  │              WARP SCHEDULERS (4)                        │ │
    │ (I$)          │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │ │
    │               │──│  │ Sched 0 │ │ Sched 1 │ │ Sched 2 │ │ Sched 3 │       │ │
    │               │  │  │ Warp 0-7│ │Warp 8-15│ │Warp16-23│ │Warp24-31│       │ │
    │               │  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │ │
    └───────────────┘  └───────┼──────────┼──────────┼──────────┼────────────────┘ │
                               │          │          │          │                   │
                               ▼          ▼          ▼          ▼                   │
    ┌──────────────────────────────────────────────────────────────────────────────┤
    │              REGISTER FILE (256KB Total, 64KB per Partition)                 │
    │                                                                              │
    │  ┌─────────────────────────────────────────────────────────────────────────┐│
    │  │ PARTITION 0 (16K Registers, 16 Banks)                                   ││
    │  │ ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐ ││
    │  │ │B0  │B1  │B2  │B3  │B4  │B5  │B6  │B7  │B8  │B9  │B10 │B11 │B12 │... │ ││
    │  │ └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘ ││
    │  └─────────────────────────────────────────────────────────────────────────┘│
    │  ┌─────────────────────────────────────────────────────────────────────────┐│
    │  │ PARTITION 1-3 (similar structure, 16K Registers each)                   ││
    │  └─────────────────────────────────────────────────────────────────────────┘│
    │                             │ (Bank Conflict Arbitration per Partition)      │
    └─────────────────────────────┼────────────────────────────────────────────────┤
                                  │                                                 │
    ┌─────────────────────────────▼────────────────────────────────────────────────┤
    │                    OPERAND COLLECTORS (Consolidators)                        │
    │  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐             │
    │  │ Collector 0      │ │ Collector 1      │ │ Collector 2      │  ...        │
    │  │ [src1][src2][src3]│ │ [src1][src2][src3]│ │ [src1][src2][src3]│            │
    │  │ ready: ■■□       │ │ ready: ■■■       │ │ ready: □□□       │             │
    │  └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘             │
    └───────────┼────────────────────┼────────────────────┼────────────────────────┤
                │                    │                    │                         │
    ┌───────────▼────────────────────▼────────────────────▼────────────────────────┤
    │                    EXECUTION UNITS (4 Partitions × 8 Cores)                  │
    │  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐    │
    │  │    PARTITION 0      │ │    PARTITION 1      │ │    PARTITION 2/3    │    │
    │  │ ┌───┬───┬───┬───┐   │ │ ┌───┬───┬───┬───┐   │ │                     │    │
    │  │ │C0 │C1 │C2 │C3 │   │ │ │C8 │C9 │C10│C11│   │ │       ...           │    │
    │  │ ├───┼───┼───┼───┤   │ │ ├───┼───┼───┼───┤   │ │                     │    │
    │  │ │C4 │C5 │C6 │C7 │   │ │ │C12│C13│C14│C15│   │ │                     │    │
    │  │ └───┴───┴───┴───┘   │ │ └───┴───┴───┴───┘   │ │                     │    │
    │  │ INT32 FP32 SFU LD/ST│ │ INT32 FP32 SFU LD/ST│ │                     │    │
    │  └─────────────────────┘ └─────────────────────┘ └─────────────────────┘    │
    └─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### Warp Scheduler

Each partition has a warp scheduler that manages 8 concurrent warps:

- **Round-robin scheduling**: Fair selection among ready warps
- **Scoreboard**: Tracks register dependencies (RAW hazards)
- **Stall detection**: Identifies warps blocked on data dependencies

Warp states:

- `INACTIVE`: Not allocated
- `READY`: Ready to issue instruction
- `EXECUTING`: Instruction in pipeline
- `STALLED_RAW`: Waiting for register write
- `STALLED_BANK`: Waiting for bank conflict resolution
- `DONE`: All instructions completed

### Register File

Per-partition register file with banking for parallel access:

- **16K registers per partition** (64KB at 32 bits each)
- **16 banks per partition** (1K registers = 4KB per bank)
- **Bank selection**: `bank_id = reg_addr % 16`
- **2-3 read ports** per partition (for operand collectors)
- **1 write port** per partition (for writeback)

**Bank conflicts** occur when multiple operands map to the same bank,
causing pipeline stalls.

### Operand Collector

Buffers operands from register file until all sources are ready:

- **2 collectors per partition** for latency hiding
- **3 operand slots** per collector (src1, src2, src3)
- **Bank arbitration**: Resolves conflicts among collectors
- **Fire when ready**: Issues to execution when all operands collected

### Execution Units

Pipelined execution with variable latency:

| Operation | Latency | Energy (pJ) |
|-----------|---------|-------------|
| INT32 ALU | 1 cycle | 1.0 |
| INT32 MUL | 4 cycles | 2.0 |
| FP32 FMA | 4 cycles | 2.0 |
| MOV | 1 cycle | 0.5 |

## Energy Model

### Per-Operation Energy (45nm baseline)

| Operation | Energy (pJ) | Notes |
|-----------|-------------|-------|
| Instruction Fetch | 10.0 | I-cache access |
| Instruction Decode | 2.0 | Decode logic |
| Warp Schedule | 2.0 | Scheduler decision |
| Register Read (32b) | 5.0 | Per bank access |
| Register Write (32b) | 5.0 | Per bank access |
| Bank Conflict | 5.0 | Additional stall cycle |
| Operand Collect | 1.0 | Per operand gathered |
| INT32 ALU | 1.0 | Add/sub/logic |
| INT32 MUL | 2.0 | Multiply |
| FP32 FMA | 2.0 | Fused multiply-add |

### Total Energy per FMA Instruction

```
Instruction overhead:
  - Fetch:           10 pJ
  - Decode:           2 pJ
  - Schedule:         2 pJ
  - Register reads:  15 pJ (3 × 5 pJ)
  - Operand collect:  3 pJ (3 × 1 pJ)
  - Register write:   5 pJ
  Subtotal:          37 pJ

Compute:
  - FMA:              2 pJ

Total:               39 pJ per FMA
Efficiency:          5.1% (2/39)
```

## Energy Comparison

### GEMM 64×64×64 Workload

| Architecture | Total Energy | pJ/MAC | Efficiency |
|--------------|--------------|--------|------------|
| **SIMT** | 10.2 µJ | 39.0 | 5.1% |
| **Systolic Array** | 524 nJ | 2.0 | 75.0% |

Systolic is 19.5× more energy efficient than SIMT.

### Conv2D 64×64×64→64, K=3 Workload

| Architecture | Total Energy | pJ/MAC | Efficiency |
|--------------|--------------|--------|------------|
| **SIMT** | 5.2 mJ | 39.0 | 5.1% |
| **Stencil Machine** | 200 µJ | 1.5 | 99.9% |

Stencil is 26× more energy efficient than SIMT.

## Why Dataflow Wins

SIMT architectures spend ~95% of energy on instruction overhead:

1. **Instruction fetch**: Every operation requires fetching from I-cache
2. **Instruction decode**: Complex decode logic for each instruction
3. **Warp scheduling**: Scheduler must select among ready warps
4. **Register file access**: 3+ banked SRAM reads per instruction
5. **Bank conflicts**: Stalls when operands collide on same bank

Dataflow architectures (systolic arrays, stencil machines) eliminate this overhead:

1. **No instructions**: Data flows directly through compute units
2. **No scheduling**: Fixed dataflow pattern, no dynamic decisions
3. **Local registers**: Only shift registers, no banked SRAM
4. **Perfect data reuse**: Each value read once, used many times

## File Structure

```
src/systars/simt/
├── __init__.py           # Module exports
├── config.py             # SIMTConfig dataclass
├── register_file.py      # Banked register file with conflicts
├── warp_scheduler.py     # Warp selection and scoreboard
├── operand_collector.py  # Operand gathering logic
├── execution_unit.py     # ALU, FMA execution pipeline
├── partition.py          # Partition combining components
├── sm_controller.py      # Top-level SM FSM
└── energy_model.py       # Energy estimation

examples/simt/
├── 01_animated_simt.py   # Terminal animation
└── 02_energy_comparison.py # SIMT vs Systolic vs Stencil
```

## Usage

### Running the Animation

```bash
python examples/simt/01_animated_simt.py --warps 4 --delay 100
```

### Running Energy Comparison Demo

```bash
python examples/simt/02_energy_comparison.py --size 64
```

### Programmatic Usage

```python
from systars.simt import SIMTConfig, SMSim, create_test_program

# Create SM
config = SIMTConfig()
sm = SMSim(config)

# Load program into warps
program = create_test_program(16)
sm.load_uniform_program(program, num_warps=4)

# Run simulation
sm.activate_warps(4)
sm.run_to_completion()

# Get statistics
stats = sm.get_statistics()
print(f"Cycles: {stats['cycles']}")
print(f"Energy: {stats['energy_pj']:.1f} pJ")
print(f"Efficiency: {stats['energy_pj']}")
```

## References

- NVIDIA CUDA Architecture documentation
- "Analyzing CUDA Workloads Using a Detailed GPU Simulator" (Bakhoda et al.)
- "Energy-Efficient Neural Networks using Data Level Parallelism" (Chen et al.)
