"""
NVIDIA Streaming Multiprocessor (SM) SIMT implementation.

This module provides an NVIDIA-style Streaming Multiprocessor (SM) implementation
based on the architecture from CUDA GPUs (Volta/Turing/Ampere era).

Architecture:
- 4 Partitions (sub-cores/processing blocks)
- 128 CUDA cores total (32 per partition)
- 64K registers total (16K per partition, 16 banks per partition)
- Operand collectors for gathering operands before execution
- Warp schedulers for SIMT execution model (32 threads per warp)

Key Differences from AMD/MasPar:
- Warp-based SIMT (32 threads execute in lockstep with predication)
- Operand collectors for register bank conflict hiding
- MSHR-based memory coalescing

Usage:
    from systars.simt.nv import SIMTConfig, SMSim

    config = SIMTConfig()
    sm = SMSim(config)

    # Load program into warps
    program = [Instruction(...), ...]
    sm.load_uniform_program(program, num_warps=4)

    # Run simulation
    sm.activate_warps(4)
    sm.run_to_completion()

    # Get statistics
    stats = sm.get_statistics()
"""

from .barrier import BarrierSlot, BarrierState, BarrierUnitSim
from .config import DEFAULT_SIMT_CONFIG, LARGE_SIMT_CONFIG, SMALL_SIMT_CONFIG, SIMTConfig
from .energy_model import (
    Conv2DWorkload,
    EnergyBreakdown,
    GEMMWorkload,
    compare_architectures,
    energy_from_sm_statistics,
    estimate_simt_energy,
    estimate_stencil_energy,
    estimate_systolic_energy,
    print_comparison,
)
from .execution_unit import OPCODE_LATENCY, OPCODE_MAP, ExecutionUnitSim, Opcode
from .global_memory import GlobalMemorySim
from .load_store_unit import AddressSpace
from .memory_coalescer import CoalescingResult, MemoryCoalescerSim, analyze_access_pattern
from .operand_collector import CollectorState, OperandCollectorSim, OperandState
from .partition import (
    PartitionSim,
    create_gemm_program,
    create_test_program,
    create_tiled_gemm_program,
    get_gemm_warp_count,
    get_warp_tile_info,
)
from .register_file import BankState, RegisterFileSim
from .shared_memory import SharedMemoryBankState, SharedMemorySim
from .sm_controller import SMSim, SMState, run_gemm_simulation
from .sm_lsu import MIOQueueEntry, MSHREntry, MSHRState, MSHRWaiter, SMLevelLSUSim
from .warp_scheduler import Instruction, SchedulingPolicy, WarpSchedulerSim, WarpState

__all__ = [
    # Config
    "SIMTConfig",
    "DEFAULT_SIMT_CONFIG",
    "SMALL_SIMT_CONFIG",
    "LARGE_SIMT_CONFIG",
    # Barrier
    "BarrierUnitSim",
    "BarrierSlot",
    "BarrierState",
    # Energy Model
    "EnergyBreakdown",
    "GEMMWorkload",
    "Conv2DWorkload",
    "estimate_simt_energy",
    "estimate_systolic_energy",
    "estimate_stencil_energy",
    "compare_architectures",
    "energy_from_sm_statistics",
    "print_comparison",
    # SM Controller
    "SMSim",
    "SMState",
    "run_gemm_simulation",
    # Partition
    "PartitionSim",
    "create_gemm_program",
    "create_test_program",
    "create_tiled_gemm_program",
    "get_gemm_warp_count",
    "get_warp_tile_info",
    # Warp Scheduler
    "WarpSchedulerSim",
    "Instruction",
    "WarpState",
    "SchedulingPolicy",
    # Register File
    "RegisterFileSim",
    "BankState",
    # Operand Collector
    "OperandCollectorSim",
    "OperandState",
    "CollectorState",
    # Execution Unit
    "ExecutionUnitSim",
    "Opcode",
    "OPCODE_MAP",
    "OPCODE_LATENCY",
    # Shared Memory
    "SharedMemorySim",
    "SharedMemoryBankState",
    # Load/Store Unit
    "AddressSpace",
    # SM-Level LSU
    "SMLevelLSUSim",
    "MSHREntry",
    "MSHRState",
    "MSHRWaiter",
    "MIOQueueEntry",
    # Global Memory
    "GlobalMemorySim",
    # Memory Coalescer
    "MemoryCoalescerSim",
    "CoalescingResult",
    "analyze_access_pattern",
]
