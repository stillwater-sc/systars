"""
Systars Configuration Module

This module defines the configuration dataclass for the systolic matmul generator.
All hardware parameters are specified here and propagate through the design.

Note: This generator produces systolic arrays for matrix multiplication (D = A × B + C),
not general systolic arrays. The dataflow modes refer to which operand stays stationary
in the processing elements.
"""

from dataclasses import dataclass
from enum import Enum, Flag, auto


class Dataflow(Flag):
    """
    Dataflow modes for the systolic matmul array.

    For the operation D = A × B + C:
    - OUTPUT_STATIONARY: Result (D) accumulates in place, A and B flow through
    - A_STATIONARY: Left operand (A) stays in PE, B and partial sums flow through
    - B_STATIONARY: Right operand (B) stays in PE, A and partial sums flow through

    Combinations (using | operator) indicate hardware supports runtime selection:
        Dataflow.OUTPUT_STATIONARY | Dataflow.B_STATIONARY  # Supports both modes

    Example:
        >>> config = SystolicConfig(dataflow=Dataflow.OUTPUT_STATIONARY)
        >>> config = SystolicConfig(dataflow=Dataflow.OUTPUT_STATIONARY | Dataflow.B_STATIONARY)
    """

    OUTPUT_STATIONARY = auto()  # D accumulates in PE, A and B flow through
    A_STATIONARY = auto()  # A stays in PE, B and partial sums flow through
    B_STATIONARY = auto()  # B stays in PE, A and partial sums flow through


class Activation(Enum):
    """Supported activation functions."""

    NONE = 0
    RELU = 1
    RELU6 = 2
    LAYERNORM = 3
    IGELU = 4
    SOFTMAX = 5


@dataclass
class SystolicConfig:
    """
    Configuration for the systolic array generator.

    This dataclass contains all parameters needed to generate a complete
    systolic array accelerator. Parameters are organized by subsystem.

    Example:
        >>> config = SystolicConfig(mesh_rows=16, mesh_cols=16)
        >>> print(config.dim)  # 16
        >>> print(config.sp_width)  # 128 (16 * 8 bits)
    """

    # =========================================================================
    # Systolic Array Dimensions
    # =========================================================================
    mesh_rows: int = 16
    """Number of tile rows in the mesh (typically 16)."""

    mesh_cols: int = 16
    """Number of tile columns in the mesh (typically 16)."""

    tile_rows: int = 1
    """Number of PE rows within each tile (typically 1)."""

    tile_cols: int = 1
    """Number of PE columns within each tile (typically 1)."""

    # =========================================================================
    # Data Types (bit widths)
    # =========================================================================
    input_bits: int = 8
    """Bit width of input activations (typically 8 for INT8)."""

    weight_bits: int = 8
    """Bit width of weights (typically 8 for INT8)."""

    acc_bits: int = 32
    """Bit width of accumulator (typically 32 for INT32)."""

    output_bits: int = 20
    """Bit width of PE output before accumulation."""

    # =========================================================================
    # Dataflow Configuration
    # =========================================================================
    dataflow: Dataflow = Dataflow.OUTPUT_STATIONARY | Dataflow.B_STATIONARY
    """
    Dataflow mode(s) for the systolic matmul array (D = A × B + C).

    Single modes:
    - OUTPUT_STATIONARY: Result accumulates in PE (best for long dot products)
    - A_STATIONARY: Left operand stays in PE (best for matvecs)
    - B_STATIONARY: Right operand stays in PE (best for weight reuse)

    Combinations (runtime selectable):
    - OUTPUT_STATIONARY | B_STATIONARY: Default, supports both modes
    - OUTPUT_STATIONARY | A_STATIONARY | B_STATIONARY: Fully configurable
    """

    # =========================================================================
    # Scratchpad Configuration
    # =========================================================================
    sp_capacity_kb: int = 256
    """Total scratchpad capacity in kilobytes."""

    sp_banks: int = 4
    """Number of scratchpad banks (more banks = more parallelism)."""

    sp_singleported: bool = True
    """If True, scratchpad banks are single-ported (saves area)."""

    spad_read_delay: int = 4
    """Read latency from scratchpad in cycles."""

    # =========================================================================
    # Accumulator Configuration
    # =========================================================================
    acc_capacity_kb: int = 64
    """Total accumulator capacity in kilobytes."""

    acc_banks: int = 2
    """Number of accumulator banks."""

    acc_sub_banks: int = 1
    """Sub-banks within each accumulator bank (for single-ported designs)."""

    acc_singleported: bool = False
    """If True, accumulator banks are single-ported."""

    acc_latency: int = 2
    """Accumulator access latency in cycles."""

    # =========================================================================
    # DMA Configuration
    # =========================================================================
    dma_maxbytes: int = 64
    """Maximum bytes per DMA transaction (typically cache line size)."""

    dma_buswidth: int = 128
    """DMA bus width in bits."""

    max_in_flight_reqs: int = 16
    """Maximum number of in-flight DMA requests."""

    # =========================================================================
    # Queue Depths
    # =========================================================================
    ld_queue_length: int = 8
    """Load instruction queue depth."""

    st_queue_length: int = 2
    """Store instruction queue depth."""

    ex_queue_length: int = 8
    """Execute instruction queue depth."""

    # =========================================================================
    # Reservation Station Configuration
    # =========================================================================
    rs_entries_ld: int = 8
    """Reservation station entries for load commands."""

    rs_entries_st: int = 4
    """Reservation station entries for store commands."""

    rs_entries_ex: int = 16
    """Reservation station entries for execute commands."""

    # =========================================================================
    # Feature Flags
    # =========================================================================
    has_training_convs: bool = True
    """Enable training convolution support (transpose convs)."""

    has_max_pool: bool = True
    """Enable max pooling during store operations."""

    has_nonlinear_activations: bool = True
    """Enable nonlinear activation functions (ReLU, etc.)."""

    has_normalizations: bool = False
    """Enable normalization operations (LayerNorm, etc.)."""

    has_first_layer_optimizations: bool = True
    """Enable first-layer optimizations (pixel repeats)."""

    # =========================================================================
    # Pipeline Tuning
    # =========================================================================
    tile_latency: int = 0
    """Pipeline latency within a tile (0 = combinational)."""

    mesh_output_delay: int = 1
    """Output delay from mesh in cycles."""

    # =========================================================================
    # Computed Properties
    # =========================================================================
    @property
    def dim(self) -> int:
        """Systolic array dimension (assumes square: mesh_rows * tile_rows)."""
        return self.mesh_rows * self.tile_rows

    @property
    def total_pes(self) -> int:
        """Total number of processing elements."""
        return self.mesh_rows * self.mesh_cols * self.tile_rows * self.tile_cols

    @property
    def sp_width(self) -> int:
        """Scratchpad row width in bits."""
        return self.mesh_cols * self.tile_cols * self.input_bits

    @property
    def sp_width_bytes(self) -> int:
        """Scratchpad row width in bytes."""
        return self.sp_width // 8

    @property
    def sp_bank_entries(self) -> int:
        """Number of rows per scratchpad bank."""
        total_bits = self.sp_capacity_kb * 1024 * 8
        return total_bits // (self.sp_banks * self.sp_width)

    @property
    def sp_total_rows(self) -> int:
        """Total number of scratchpad rows across all banks."""
        return self.sp_banks * self.sp_bank_entries

    @property
    def acc_width(self) -> int:
        """Accumulator row width in bits."""
        return self.mesh_cols * self.tile_cols * self.acc_bits

    @property
    def acc_width_bytes(self) -> int:
        """Accumulator row width in bytes."""
        return self.acc_width // 8

    @property
    def acc_bank_entries(self) -> int:
        """Number of rows per accumulator bank."""
        total_bits = self.acc_capacity_kb * 1024 * 8
        return total_bits // (self.acc_banks * self.acc_width)

    @property
    def acc_total_rows(self) -> int:
        """Total number of accumulator rows across all banks."""
        return self.acc_banks * self.acc_bank_entries

    @property
    def sp_addr_bits(self) -> int:
        """Bits needed to address all scratchpad rows."""
        return (self.sp_total_rows - 1).bit_length()

    @property
    def acc_addr_bits(self) -> int:
        """Bits needed to address all accumulator rows."""
        return (self.acc_total_rows - 1).bit_length()

    @property
    def local_addr_bits(self) -> int:
        """Bits needed for unified local address (max of sp and acc)."""
        return max(self.sp_addr_bits, self.acc_addr_bits)

    @property
    def max_block_len(self) -> int:
        """Maximum block length for scratchpad transfers."""
        return self.dma_maxbytes // self.sp_width_bytes

    @property
    def max_block_len_acc(self) -> int:
        """Maximum block length for accumulator transfers."""
        return self.dma_maxbytes // self.acc_width_bytes

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.mesh_rows > 0, "mesh_rows must be positive"
        assert self.mesh_cols > 0, "mesh_cols must be positive"
        assert self.tile_rows > 0, "tile_rows must be positive"
        assert self.tile_cols > 0, "tile_cols must be positive"
        assert self.input_bits > 0, "input_bits must be positive"
        assert self.weight_bits > 0, "weight_bits must be positive"
        assert self.acc_bits >= self.input_bits + self.weight_bits, (
            "acc_bits should be >= input_bits + weight_bits to avoid overflow"
        )
        assert self.sp_banks > 0, "sp_banks must be positive"
        assert self.acc_banks > 0, "acc_banks must be positive"
        assert self.sp_capacity_kb > 0, "sp_capacity_kb must be positive"
        assert self.acc_capacity_kb > 0, "acc_capacity_kb must be positive"


# Pre-defined configurations
DEFAULT_CONFIG = SystolicConfig()
"""Default configuration."""

LEAN_CONFIG = SystolicConfig(
    dataflow=Dataflow.B_STATIONARY,
    max_in_flight_reqs=64,
    has_normalizations=False,
)
"""Lean configuration for faster simulation."""

CHIP_CONFIG = SystolicConfig(
    sp_capacity_kb=64,
    acc_capacity_kb=32,
    dataflow=Dataflow.B_STATIONARY,
    acc_singleported=True,
    acc_sub_banks=2,
    mesh_output_delay=2,
)
"""Configuration optimized for tapeout."""

LARGE_CONFIG = SystolicConfig(
    mesh_rows=32,
    mesh_cols=32,
    sp_capacity_kb=512,
    acc_capacity_kb=128,
)
"""Large configuration for high-performance applications."""
