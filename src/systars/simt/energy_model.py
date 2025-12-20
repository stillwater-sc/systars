"""
Energy Model for SIMT vs Systolic vs Stencil Comparison.

This module provides energy estimation for comparing different accelerator
architectures on the same workload (e.g., GEMM, Conv2D).

The key insight is that SIMT architectures spend significant energy on
instruction overhead (fetch, decode, schedule, register file access)
while systolic arrays and stencil machines achieve near-optimal energy
efficiency by eliminating instruction overhead through dataflow execution.

Energy Breakdown (45nm baseline):
    SIMT per operation:
        - Instruction fetch: 10 pJ
        - Instruction decode: 2 pJ
        - Warp scheduling: 2 pJ
        - Register reads (3×): 15 pJ
        - ALU/FMA: 1-2 pJ
        - Register write: 5 pJ
        Total: ~35 pJ per FMA

    Systolic Array per operation:
        - Data movement (amortized): 2-5 pJ
        - MAC operation: 1-2 pJ
        Total: ~3-7 pJ per FMA

    Stencil Machine per operation:
        - Line buffer read (amortized): 1-2 pJ
        - MAC operation: 1-2 pJ
        Total: ~2-4 pJ per FMA

Efficiency Comparison:
    - SIMT: ~5-10% of energy goes to useful compute
    - Systolic: ~50-70% of energy goes to useful compute
    - Stencil: ~60-80% of energy goes to useful compute
"""

from dataclasses import dataclass

from .config import SIMTConfig


@dataclass
class EnergyBreakdown:
    """Breakdown of energy consumption by category."""

    # Instruction overhead
    instruction_fetch_pj: float = 0.0
    instruction_decode_pj: float = 0.0
    scheduling_pj: float = 0.0

    # Data movement
    register_read_pj: float = 0.0
    register_write_pj: float = 0.0
    bank_conflict_pj: float = 0.0
    operand_collect_pj: float = 0.0

    # Compute
    alu_pj: float = 0.0
    fma_pj: float = 0.0
    mul_pj: float = 0.0

    # Memory (if applicable)
    shared_mem_pj: float = 0.0
    l1_cache_pj: float = 0.0
    dram_pj: float = 0.0

    @property
    def instruction_overhead_pj(self) -> float:
        """Total instruction overhead energy."""
        return self.instruction_fetch_pj + self.instruction_decode_pj + self.scheduling_pj

    @property
    def data_movement_pj(self) -> float:
        """Total data movement energy."""
        return (
            self.register_read_pj
            + self.register_write_pj
            + self.bank_conflict_pj
            + self.operand_collect_pj
        )

    @property
    def compute_pj(self) -> float:
        """Total compute energy (useful work)."""
        return self.alu_pj + self.fma_pj + self.mul_pj

    @property
    def total_pj(self) -> float:
        """Total energy consumption."""
        return (
            self.instruction_overhead_pj
            + self.data_movement_pj
            + self.compute_pj
            + self.shared_mem_pj
            + self.l1_cache_pj
            + self.dram_pj
        )

    @property
    def efficiency_percent(self) -> float:
        """Percentage of energy that goes to useful compute."""
        if self.total_pj == 0:
            return 0.0
        return (self.compute_pj / self.total_pj) * 100

    @property
    def memory_pj(self) -> float:
        """Total memory access energy."""
        return self.shared_mem_pj + self.l1_cache_pj + self.dram_pj

    def __str__(self) -> str:
        """Format energy breakdown as string."""
        lines = [
            "Energy Breakdown:",
            f"  Instruction Overhead: {self.instruction_overhead_pj:.1f} pJ",
            f"    - Fetch:    {self.instruction_fetch_pj:.1f} pJ",
            f"    - Decode:   {self.instruction_decode_pj:.1f} pJ",
            f"    - Schedule: {self.scheduling_pj:.1f} pJ",
            f"  Data Movement:        {self.data_movement_pj:.1f} pJ",
            f"    - Reg Read:  {self.register_read_pj:.1f} pJ",
            f"    - Reg Write: {self.register_write_pj:.1f} pJ",
            f"    - Conflicts: {self.bank_conflict_pj:.1f} pJ",
            f"  Memory:               {self.memory_pj:.1f} pJ",
            f"    - Shared:  {self.shared_mem_pj:.1f} pJ",
            f"    - L1 Cache: {self.l1_cache_pj:.1f} pJ",
            f"    - DRAM:    {self.dram_pj:.1f} pJ",
            f"  Compute:              {self.compute_pj:.1f} pJ",
            f"    - ALU:  {self.alu_pj:.1f} pJ",
            f"    - FMA:  {self.fma_pj:.1f} pJ",
            f"  Total:                {self.total_pj:.1f} pJ",
            f"  Efficiency:           {self.efficiency_percent:.1f}%",
        ]
        return "\n".join(lines)


@dataclass
class GEMMWorkload:
    """GEMM workload specification."""

    M: int  # Output rows
    N: int  # Output columns
    K: int  # Reduction dimension

    @property
    def total_macs(self) -> int:
        """Total MAC operations for GEMM."""
        return self.M * self.N * self.K

    @property
    def a_elements(self) -> int:
        """Elements in matrix A."""
        return self.M * self.K

    @property
    def b_elements(self) -> int:
        """Elements in matrix B."""
        return self.K * self.N

    @property
    def c_elements(self) -> int:
        """Elements in output matrix C."""
        return self.M * self.N


@dataclass
class Conv2DWorkload:
    """Conv2D workload specification."""

    input_height: int
    input_width: int
    input_channels: int
    output_channels: int
    kernel_size: int
    stride: int = 1

    @property
    def output_height(self) -> int:
        """Output feature map height."""
        return (self.input_height - self.kernel_size) // self.stride + 1

    @property
    def output_width(self) -> int:
        """Output feature map width."""
        return (self.input_width - self.kernel_size) // self.stride + 1

    @property
    def total_macs(self) -> int:
        """Total MAC operations for Conv2D."""
        return (
            self.output_height
            * self.output_width
            * self.output_channels
            * self.input_channels
            * self.kernel_size
            * self.kernel_size
        )


def estimate_simt_energy(
    workload: GEMMWorkload,
    config: SIMTConfig,
    include_memory: bool = True,
) -> EnergyBreakdown:
    """
    Estimate SIMT energy for GEMM workload.

    In SIMT, each MAC requires:
    - 1 instruction (FMA)
    - 3 register reads (A, B, accumulator)
    - 1 register write (accumulator update)

    Memory operations (if include_memory=True):
    - Load A and B elements from global memory
    - Store C elements to global memory
    - Shared memory for tile reuse

    Args:
        workload: GEMM workload specification
        config: SIMT configuration
        include_memory: Include memory access energy (default True)

    Returns:
        Energy breakdown.
    """
    num_macs = workload.total_macs

    # Each MAC = 1 FMA instruction
    num_instructions = num_macs

    # Memory operations (simplified tiled GEMM model)
    # Assume tiles load A and B once, with some reuse
    # Shared memory for tile coordination
    shared_mem_accesses = 0
    dram_accesses = 0

    if include_memory:
        # Each input element read once from DRAM, cached in shared memory
        # A: M×K elements, B: K×N elements, C: M×N elements (write)
        dram_reads = workload.a_elements + workload.b_elements
        dram_writes = workload.c_elements
        dram_accesses = dram_reads + dram_writes

        # Shared memory used for tile coordination
        # Assume 16×16 tiles, each tile loads to shared memory
        tile_size = 16
        num_tiles_m = (workload.M + tile_size - 1) // tile_size
        num_tiles_n = (workload.N + tile_size - 1) // tile_size
        num_tiles_k = (workload.K + tile_size - 1) // tile_size
        shared_mem_accesses = num_tiles_m * num_tiles_n * num_tiles_k * tile_size * tile_size * 2

    breakdown = EnergyBreakdown(
        instruction_fetch_pj=num_instructions * config.instruction_fetch_energy_pj,
        instruction_decode_pj=num_instructions * config.instruction_decode_energy_pj,
        scheduling_pj=num_instructions * config.scheduler_energy_pj,
        register_read_pj=num_instructions * 3 * config.register_read_energy_pj,
        register_write_pj=num_instructions * config.register_write_energy_pj,
        operand_collect_pj=num_instructions * 3 * config.operand_collect_energy_pj,
        fma_pj=num_instructions * config.fma_energy_pj,
        shared_mem_pj=shared_mem_accesses * config.shared_mem_access_energy_pj,
        dram_pj=dram_accesses * config.dram_access_energy_pj,
    )

    return breakdown


def energy_from_sm_statistics(stats: dict) -> EnergyBreakdown:
    """
    Create EnergyBreakdown from SMSim statistics.

    Args:
        stats: Statistics dictionary from SMSim.get_statistics()

    Returns:
        EnergyBreakdown populated from simulation data.
    """
    # Aggregate partition stats
    total_shared_mem_energy = 0.0
    total_dram_energy = 0.0
    total_coalescing_energy = 0.0

    for p_stats in stats.get("partition_stats", []):
        lsu_stats = p_stats.get("lsu_stats", {})
        smem_stats = p_stats.get("shared_memory_stats", {})

        total_shared_mem_energy += smem_stats.get("total_energy_pj", 0.0)
        total_dram_energy += lsu_stats.get("total_energy_pj", 0.0)

        coalescer_stats = lsu_stats.get("coalescer_stats", {})
        total_coalescing_energy += coalescer_stats.get("total_energy_pj", 0.0)

    return EnergyBreakdown(
        # Total energy is reported by SMSim
        fma_pj=stats.get("energy_pj", 0.0),  # Simplified: total as FMA
        shared_mem_pj=total_shared_mem_energy,
        dram_pj=total_dram_energy + total_coalescing_energy,
    )


def estimate_systolic_energy(
    workload: GEMMWorkload,
    _input_bits: int = 8,
    _acc_bits: int = 32,
    mac_energy_pj: float = 1.5,
    reg_energy_pj: float = 0.5,
) -> EnergyBreakdown:
    """
    Estimate systolic array energy for GEMM workload.

    In a systolic array, each MAC requires:
    - MAC operation
    - Local register access (shift registers)
    - NO instruction overhead

    Data is loaded once and reused across the array.

    Args:
        workload: GEMM workload specification
        input_bits: Input element width
        acc_bits: Accumulator width
        mac_energy_pj: Energy per MAC operation
        reg_energy_pj: Energy per register access

    Returns:
        Energy breakdown.
    """
    num_macs = workload.total_macs

    # Systolic array: data flows through, minimal register access
    # Each PE has local registers that shift data

    breakdown = EnergyBreakdown(
        # No instruction overhead
        instruction_fetch_pj=0,
        instruction_decode_pj=0,
        scheduling_pj=0,
        # Minimal data movement (local register shifts)
        register_read_pj=num_macs * reg_energy_pj * 0.5,  # Amortized
        register_write_pj=num_macs * reg_energy_pj * 0.5,
        # All energy goes to compute
        fma_pj=num_macs * mac_energy_pj,
    )

    return breakdown


def estimate_stencil_energy(
    workload: Conv2DWorkload,
    _input_bits: int = 8,
    _acc_bits: int = 32,
    mac_energy_pj: float = 1.5,
    line_buffer_energy_pj: float = 0.3,
) -> EnergyBreakdown:
    """
    Estimate stencil machine energy for Conv2D workload.

    In a stencil machine:
    - Line buffers store K_h rows for window extraction
    - Each input pixel is read once from DRAM
    - Window former extracts K_h × K_w pixels
    - MAC array computes all output channels

    Args:
        workload: Conv2D workload specification
        input_bits: Input element width
        acc_bits: Accumulator width
        mac_energy_pj: Energy per MAC operation
        line_buffer_energy_pj: Energy per line buffer access

    Returns:
        Energy breakdown.
    """
    num_macs = workload.total_macs

    # Stencil machine: 1× DRAM read per input pixel
    input_pixels = workload.input_height * workload.input_width * workload.input_channels

    # Line buffer accesses (amortized across kernel reuse)
    line_buffer_accesses = input_pixels * workload.kernel_size  # Per row

    breakdown = EnergyBreakdown(
        # No instruction overhead
        instruction_fetch_pj=0,
        instruction_decode_pj=0,
        scheduling_pj=0,
        # Line buffer data movement (highly efficient)
        register_read_pj=line_buffer_accesses * line_buffer_energy_pj,
        # Compute
        fma_pj=num_macs * mac_energy_pj,
    )

    return breakdown


def compare_architectures(
    gemm: GEMMWorkload | None = None,
    conv2d: Conv2DWorkload | None = None,
) -> dict:
    """
    Compare energy efficiency of SIMT, Systolic, and Stencil architectures.

    Args:
        gemm: GEMM workload (if provided)
        conv2d: Conv2D workload (if provided)

    Returns:
        Comparison results.
    """
    results = {}

    if gemm:
        config = SIMTConfig()

        simt_energy = estimate_simt_energy(gemm, config)
        systolic_energy = estimate_systolic_energy(gemm)

        results["gemm"] = {
            "workload": f"GEMM {gemm.M}×{gemm.N}×{gemm.K}",
            "total_macs": gemm.total_macs,
            "simt": {
                "total_pj": simt_energy.total_pj,
                "efficiency": simt_energy.efficiency_percent,
                "pj_per_mac": simt_energy.total_pj / gemm.total_macs,
            },
            "systolic": {
                "total_pj": systolic_energy.total_pj,
                "efficiency": systolic_energy.efficiency_percent,
                "pj_per_mac": systolic_energy.total_pj / gemm.total_macs,
            },
            "speedup": simt_energy.total_pj / max(1, systolic_energy.total_pj),
        }

    if conv2d:
        stencil_energy = estimate_stencil_energy(conv2d)

        # Also compute SIMT energy for conv2d (treat as flattened GEMM)
        gemm_equiv = GEMMWorkload(
            M=conv2d.output_height * conv2d.output_width,
            N=conv2d.output_channels,
            K=conv2d.input_channels * conv2d.kernel_size * conv2d.kernel_size,
        )
        config = SIMTConfig()
        simt_conv_energy = estimate_simt_energy(gemm_equiv, config)

        results["conv2d"] = {
            "workload": (
                f"Conv2D {conv2d.input_height}×{conv2d.input_width}×"
                f"{conv2d.input_channels} → {conv2d.output_channels}, "
                f"K={conv2d.kernel_size}"
            ),
            "total_macs": conv2d.total_macs,
            "simt": {
                "total_pj": simt_conv_energy.total_pj,
                "efficiency": simt_conv_energy.efficiency_percent,
                "pj_per_mac": simt_conv_energy.total_pj / conv2d.total_macs,
            },
            "stencil": {
                "total_pj": stencil_energy.total_pj,
                "efficiency": stencil_energy.efficiency_percent,
                "pj_per_mac": stencil_energy.total_pj / conv2d.total_macs,
            },
            "speedup": simt_conv_energy.total_pj / max(1, stencil_energy.total_pj),
        }

    return results


def print_comparison(results: dict) -> None:
    """Print comparison results in formatted table."""
    print("=" * 80)
    print("ENERGY EFFICIENCY COMPARISON: SIMT vs Systolic/Stencil")
    print("=" * 80)
    print()

    if "gemm" in results:
        r = results["gemm"]
        print(f"Workload: {r['workload']}")
        print(f"Total MACs: {r['total_macs']:,}")
        print()
        print(f"{'Architecture':<15} {'Total Energy':<15} {'pJ/MAC':<10} {'Efficiency':<12}")
        print("-" * 52)
        print(
            f"{'SIMT':<15} {r['simt']['total_pj'] / 1e3:>10.1f} nJ "
            f"{r['simt']['pj_per_mac']:>8.1f}  "
            f"{r['simt']['efficiency']:>8.1f}%"
        )
        print(
            f"{'Systolic':<15} {r['systolic']['total_pj'] / 1e3:>10.1f} nJ "
            f"{r['systolic']['pj_per_mac']:>8.1f}  "
            f"{r['systolic']['efficiency']:>8.1f}%"
        )
        print()
        print(f"Systolic is {r['speedup']:.1f}× more energy efficient than SIMT")
        print()

    if "conv2d" in results:
        r = results["conv2d"]
        print(f"Workload: {r['workload']}")
        print(f"Total MACs: {r['total_macs']:,}")
        print()
        print(f"{'Architecture':<15} {'Total Energy':<15} {'pJ/MAC':<10} {'Efficiency':<12}")
        print("-" * 52)
        print(
            f"{'SIMT':<15} {r['simt']['total_pj'] / 1e3:>10.1f} nJ "
            f"{r['simt']['pj_per_mac']:>8.1f}  "
            f"{r['simt']['efficiency']:>8.1f}%"
        )
        print(
            f"{'Stencil':<15} {r['stencil']['total_pj'] / 1e3:>10.1f} nJ "
            f"{r['stencil']['pj_per_mac']:>8.1f}  "
            f"{r['stencil']['efficiency']:>8.1f}%"
        )
        print()
        print(f"Stencil is {r['speedup']:.1f}× more energy efficient than SIMT")

    print()
    print("=" * 80)
