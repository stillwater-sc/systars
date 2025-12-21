"""
MasPar MP-1/MP-2 configuration.

This module defines the configuration dataclass for the MasPar SIMD
array processor simulator, based on the MP-2 architecture specifications.

Architecture Reference:
- MP-2: 32-bit ALU, 64 registers per PE, 64KB local memory
- Array sizes: 64x64 (4096 PEs) to 128x128 (16384 PEs)
- XNET: 8-neighbor mesh with toroidal wrap-around
- Global router: Arbitrary PE-to-PE communication
"""

from dataclasses import dataclass


@dataclass
class MasParConfig:
    """
    Configuration for MasPar SIMD array processor.

    Default values match the MP-2 architecture with a 64x64 PE array.
    """

    # Array dimensions
    array_rows: int = 64  # Number of PE rows
    array_cols: int = 64  # Number of PE columns

    # PE specifications (MP-2)
    registers_per_pe: int = 64  # 64 x 32-bit registers per PE
    register_width: int = 32  # Bits per register
    local_memory_kb: int = 64  # KB of local SRAM per PE

    # ALU configuration
    alu_width: int = 32  # 32-bit ALU (MP-2), 4-bit for MP-1
    has_fpu: bool = True  # Floating-point unit support

    # XNET mesh interconnect
    xnet_latency: int = 1  # Cycles for neighbor-to-neighbor transfer

    # Global router
    router_latency: int = 10  # Cycles for arbitrary PE-to-PE routing

    # Instruction timing (cycles)
    alu_int_latency: int = 1  # Integer ALU operations
    alu_mul_latency: int = 4  # Integer multiply
    fpu_latency: int = 4  # Floating-point operations
    memory_latency: int = 2  # Local memory access

    # Energy model (pJ per operation) - estimates
    alu_energy_pj: float = 1.0
    mul_energy_pj: float = 2.0
    fpu_energy_pj: float = 3.0
    register_read_energy_pj: float = 0.5
    register_write_energy_pj: float = 0.5
    local_memory_read_energy_pj: float = 5.0
    local_memory_write_energy_pj: float = 5.0
    xnet_transfer_energy_pj: float = 2.0
    router_transfer_energy_pj: float = 10.0

    @property
    def total_pes(self) -> int:
        """Total number of processing elements."""
        return self.array_rows * self.array_cols

    @property
    def local_memory_bytes(self) -> int:
        """Local memory size per PE in bytes."""
        return self.local_memory_kb * 1024

    @property
    def total_registers(self) -> int:
        """Total registers across all PEs."""
        return self.total_pes * self.registers_per_pe

    @property
    def total_local_memory_kb(self) -> int:
        """Total local memory across all PEs in KB."""
        return self.total_pes * self.local_memory_kb

    @property
    def total_local_memory_mb(self) -> int:
        """Total local memory across all PEs in MB."""
        return self.total_local_memory_kb // 1024


# Pre-defined configurations
DEFAULT_MASPAR_CONFIG = MasParConfig()  # 64x64 MP-2

SMALL_MASPAR_CONFIG = MasParConfig(
    array_rows=4,
    array_cols=4,
)

MEDIUM_MASPAR_CONFIG = MasParConfig(
    array_rows=16,
    array_cols=16,
)

LARGE_MASPAR_CONFIG = MasParConfig(
    array_rows=128,
    array_cols=128,
)

# MP-1 configuration (4-bit ALU, fewer registers)
MP1_CONFIG = MasParConfig(
    array_rows=64,
    array_cols=64,
    registers_per_pe=48,  # MP-1 had 48 registers
    alu_width=4,  # 4-bit ALU (8 cycles for 32-bit ops)
    alu_int_latency=8,  # 8 cycles for 32-bit integer ops
    alu_mul_latency=32,  # 32 cycles for 32-bit multiply
)
