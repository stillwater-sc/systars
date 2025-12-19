"""
SIMT Configuration.

Configuration dataclass for the NVIDIA-style Streaming Multiprocessor.
"""

from dataclasses import dataclass


@dataclass
class SIMTConfig:
    """
    Configuration for SIMT Streaming Multiprocessor.

    Based on NVIDIA SM architecture with:
    - 4 partitions (processing blocks)
    - 32 cores (8 per partition)
    - 64K registers (16K per partition)
    - Operand collectors for register gathering
    - Warp-based SIMT execution

    Energy Model:
        All energy values are in picojoules (pJ) at 45nm baseline.
        Scale by (45nm/target_nm)^2 for other process nodes.
    """

    # =========================================================================
    # SM Structure
    # =========================================================================

    num_partitions: int = 4
    """Number of processing partitions (sub-cores) in the SM."""

    cores_per_partition: int = 8
    """Number of CUDA cores per partition (32 total)."""

    warp_size: int = 32
    """Number of threads per warp (SIMT execution width)."""

    max_warps_per_partition: int = 8
    """Maximum concurrent warps per partition."""

    max_warps_per_sm: int = 32
    """Maximum concurrent warps per SM (4 partitions × 8 warps)."""

    # =========================================================================
    # Register File (per partition)
    # =========================================================================

    registers_per_partition: int = 16384
    """Number of 32-bit registers per partition (16K)."""

    register_banks_per_partition: int = 16
    """Number of register file banks per partition."""

    registers_per_bank: int = 1024
    """Number of registers per bank (1K registers = 4KB)."""

    register_width: int = 32
    """Width of each register in bits."""

    # Derived values (computed properties)
    # - Total registers: 4 × 16K = 64K registers
    # - Total capacity: 64K × 4B = 256KB
    # - Per bank: 1K × 4B = 4KB

    # =========================================================================
    # Operand Collectors (per partition)
    # =========================================================================

    collectors_per_partition: int = 2
    """Number of operand collectors per partition for latency hiding."""

    operands_per_collector: int = 3
    """Number of source operands per instruction (src1, src2, src3)."""

    # =========================================================================
    # Instruction Configuration
    # =========================================================================

    instruction_bits: int = 64
    """Instruction width in bits (SASS-style encoding)."""

    max_instructions: int = 1024
    """Maximum instructions in I-cache."""

    # =========================================================================
    # Memory Configuration
    # =========================================================================

    shared_memory_kb: int = 48
    """Shared memory size in KB (configurable partition with L1)."""

    l1_cache_kb: int = 16
    """L1 data cache size in KB."""

    # =========================================================================
    # Pipeline Configuration
    # =========================================================================

    pipeline_stages: int = 4
    """Number of pipeline stages: FETCH, DECODE, EXECUTE, WRITEBACK."""

    alu_latency: int = 1
    """Latency of integer ALU operations in cycles."""

    fma_latency: int = 4
    """Latency of FP32 FMA operations in cycles."""

    # =========================================================================
    # Energy Model Parameters (45nm baseline, in pJ)
    # =========================================================================

    register_read_energy_pj: float = 5.0
    """Energy per 32-bit register read in picojoules."""

    register_write_energy_pj: float = 5.0
    """Energy per 32-bit register write in picojoules."""

    bank_conflict_energy_pj: float = 5.0
    """Additional energy per bank conflict replay cycle."""

    alu_energy_pj: float = 1.0
    """Energy per INT32 ALU operation (add/sub/logic)."""

    mul_energy_pj: float = 2.0
    """Energy per INT32 multiply operation."""

    fma_energy_pj: float = 2.0
    """Energy per FP32 fused multiply-add operation."""

    instruction_fetch_energy_pj: float = 10.0
    """Energy per instruction fetch from I-cache."""

    instruction_decode_energy_pj: float = 2.0
    """Energy per instruction decode."""

    scheduler_energy_pj: float = 2.0
    """Energy per warp scheduling decision."""

    operand_collect_energy_pj: float = 1.0
    """Energy per operand collected from register file."""

    shared_mem_access_energy_pj: float = 8.0
    """Energy per 32-bit shared memory access."""

    l1_cache_access_energy_pj: float = 15.0
    """Energy per L1 cache line access."""

    # =========================================================================
    # Derived Properties
    # =========================================================================

    @property
    def total_cores(self) -> int:
        """Total number of CUDA cores in the SM."""
        return self.num_partitions * self.cores_per_partition

    @property
    def total_registers(self) -> int:
        """Total number of registers in the SM."""
        return self.num_partitions * self.registers_per_partition

    @property
    def register_file_kb(self) -> int:
        """Total register file size in KB."""
        return (self.total_registers * self.register_width) // (8 * 1024)

    @property
    def registers_per_thread(self) -> int:
        """Maximum registers per thread (hardware limit is typically 255)."""
        return min(255, self.registers_per_partition // self.warp_size)

    def instruction_energy_pj(self) -> float:
        """Total energy overhead per instruction (excluding ALU)."""
        return (
            self.instruction_fetch_energy_pj
            + self.instruction_decode_energy_pj
            + self.scheduler_energy_pj
            + self.operands_per_collector * self.operand_collect_energy_pj
            + self.operands_per_collector * self.register_read_energy_pj
            + self.register_write_energy_pj  # Destination write
        )

    def fma_total_energy_pj(self) -> float:
        """Total energy per FMA operation including instruction overhead."""
        return self.instruction_energy_pj() + self.fma_energy_pj

    def alu_total_energy_pj(self) -> float:
        """Total energy per ALU operation including instruction overhead."""
        return self.instruction_energy_pj() + self.alu_energy_pj


# Pre-defined configurations
DEFAULT_SIMT_CONFIG = SIMTConfig()

# Smaller config for faster simulation
SMALL_SIMT_CONFIG = SIMTConfig(
    num_partitions=2,
    cores_per_partition=4,
    max_warps_per_partition=4,
    max_warps_per_sm=8,
    registers_per_partition=4096,
    registers_per_bank=256,
)

# Large config matching high-end GPUs
LARGE_SIMT_CONFIG = SIMTConfig(
    num_partitions=4,
    cores_per_partition=16,  # 64 cores total
    max_warps_per_partition=16,
    max_warps_per_sm=64,
    registers_per_partition=32768,  # 32K per partition
    registers_per_bank=2048,
)
