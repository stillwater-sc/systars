"""
SIMT Streaming Multiprocessor module.

This module provides an NVIDIA-style Streaming Multiprocessor (SM) implementation
for comparing energy efficiency against systolic arrays and stencil machines.

Architecture:
- 4 Partitions (sub-cores/processing blocks)
- 32 CUDA cores total (8 per partition)
- 64K registers total (16K per partition, 16 banks per partition)
- Operand collectors for gathering operands before execution
- Warp schedulers for SIMT execution model

Usage:
    from systars.simt import SIMTConfig, SMSim

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

from .config import DEFAULT_SIMT_CONFIG, LARGE_SIMT_CONFIG, SMALL_SIMT_CONFIG, SIMTConfig
from .energy_model import (
    Conv2DWorkload,
    EnergyBreakdown,
    GEMMWorkload,
    compare_architectures,
    estimate_simt_energy,
    estimate_stencil_energy,
    estimate_systolic_energy,
)
from .execution_unit import OPCODE_LATENCY, OPCODE_MAP, ExecutionUnitSim, Opcode
from .operand_collector import CollectorState, OperandCollectorSim, OperandState
from .partition import PartitionSim, create_gemm_program, create_test_program
from .register_file import BankState, RegisterFileSim
from .sm_controller import SMSim, SMState, run_gemm_simulation
from .warp_scheduler import Instruction, SchedulingPolicy, WarpSchedulerSim, WarpState

__all__ = [
    # Config
    "SIMTConfig",
    "DEFAULT_SIMT_CONFIG",
    "SMALL_SIMT_CONFIG",
    "LARGE_SIMT_CONFIG",
    # Energy Model
    "EnergyBreakdown",
    "GEMMWorkload",
    "Conv2DWorkload",
    "estimate_simt_energy",
    "estimate_systolic_energy",
    "estimate_stencil_energy",
    "compare_architectures",
    # SM Controller
    "SMSim",
    "SMState",
    "run_gemm_simulation",
    # Partition
    "PartitionSim",
    "create_gemm_program",
    "create_test_program",
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
]
