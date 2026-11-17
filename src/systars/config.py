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
        >>> config = SystolicConfig(grid_rows=16, grid_cols=16)
        >>> print(config.dim)  # 16
        >>> print(config.sp_width)  # 128 (16 * 8 bits)
    """

    # =========================================================================
    # Systolic Array Dimensions
    # =========================================================================
    grid_rows: int = 16
    """Number of PEArray rows in the SystolicArray grid (typically 16)."""

    grid_cols: int = 16
    """Number of PEArray columns in the SystolicArray grid (typically 16)."""

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
    """Accumulate width of the MAC (typically 32 for INT32); also the Combiner's element width."""

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

    spad_read_delay: int = 4
    """Read latency from scratchpad in cycles."""

    # =========================================================================
    # Combiner Configuration
    # =========================================================================
    combiner_capacity_kb: int = 64
    """Total Combiner capacity in kilobytes."""

    combiner_banks: int = 2
    """Number of Combiner banks."""

    combiner_latency: int = 2
    """Combiner access latency in cycles."""

    # =========================================================================
    # BlockMover Configuration
    # =========================================================================
    blockmover_buswidth: int = 0
    """BlockMover bus width in bits. 0 (default) = automatic: the widest scratchpad or
    Combiner row, at least 128 bits, rounded up to a power of two. The BlockMover
    moves one row per beat, so an explicit width must be at least that wide."""

    # =========================================================================
    # Queue Depths
    # =========================================================================
    ld_queue_length: int = 8
    """Depth of the BlockMover's L3 -> L2 (LOAD) command queue in SystolicTop."""

    st_queue_length: int = 2
    """Depth of the BlockMover's Combiner -> L3 (STORE) command queue."""

    ex_queue_length: int = 8
    """Depth of the Streamer's (execute) command queue."""

    # =========================================================================
    # Tile Buffers (credits and tile-ready tokens, controller/sync.py)
    # =========================================================================
    l2_slots: int = 4
    """Tile slots in the scratchpad (L2), one credit each. Two per operand (A and
    B) double-buffers. A slot is a tag: commands carry its address explicitly."""

    combiner_slots: int = 2
    """Tile slots in the Combiner, one credit each."""

    # =========================================================================
    # Computed Properties
    # =========================================================================
    @property
    def dim(self) -> int:
        """Systolic array dimension (assumes square: grid_rows * tile_rows)."""
        return self.grid_rows * self.tile_rows

    @property
    def total_pes(self) -> int:
        """Total number of processing elements."""
        return self.grid_rows * self.grid_cols * self.tile_rows * self.tile_cols

    @property
    def sp_width(self) -> int:
        """Scratchpad row width in bits."""
        return self.grid_cols * self.tile_cols * self.input_bits

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
    def combiner_width(self) -> int:
        """Combiner row width in bits."""
        return self.grid_cols * self.tile_cols * self.acc_bits

    @property
    def combiner_width_bytes(self) -> int:
        """Combiner row width in bytes."""
        return self.combiner_width // 8

    @property
    def combiner_bank_entries(self) -> int:
        """Number of rows per Combiner bank."""
        total_bits = self.combiner_capacity_kb * 1024 * 8
        return total_bits // (self.combiner_banks * self.combiner_width)

    @property
    def combiner_total_rows(self) -> int:
        """Total number of Combiner rows across all banks."""
        return self.combiner_banks * self.combiner_bank_entries

    @property
    def sp_addr_bits(self) -> int:
        """Bits needed to address all scratchpad rows."""
        return (self.sp_total_rows - 1).bit_length()

    @property
    def combiner_addr_bits(self) -> int:
        """Bits needed to address all Combiner rows."""
        return (self.combiner_total_rows - 1).bit_length()

    @property
    def local_addr_bits(self) -> int:
        """Bits needed for unified local address (max of sp and acc)."""
        return max(self.sp_addr_bits, self.combiner_addr_bits)

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.blockmover_buswidth == 0:
            widest = max(128, self.sp_width, self.combiner_width)
            self.blockmover_buswidth = 1 << (widest - 1).bit_length()
        assert self.grid_rows > 0, "grid_rows must be positive"
        assert self.grid_cols > 0, "grid_cols must be positive"
        assert self.tile_rows > 0, "tile_rows must be positive"
        assert self.tile_cols > 0, "tile_cols must be positive"
        assert self.input_bits > 0, "input_bits must be positive"
        assert self.weight_bits > 0, "weight_bits must be positive"
        assert self.acc_bits >= self.input_bits + self.weight_bits, (
            "acc_bits should be >= input_bits + weight_bits to avoid overflow"
        )
        # Banks are interleaved on the low address bits (LocalAddr), so consecutive
        # rows walk the banks: a count that is not a power of two would map some
        # addresses to banks that do not exist
        for name in ("sp_banks", "combiner_banks"):
            banks = getattr(self, name)
            assert banks > 0 and banks & (banks - 1) == 0, (
                f"{name} must be a power of two (banks are the low address bits), got {banks}"
            )
        assert self.sp_capacity_kb > 0, "sp_capacity_kb must be positive"
        assert self.combiner_capacity_kb > 0, "combiner_capacity_kb must be positive"
        # Slot tags are 4-bit fields of cmd_sync (controller/sync.py)
        for name in ("l2_slots", "combiner_slots"):
            slots = getattr(self, name)
            assert 1 <= slots <= 16, f"{name} must be 1..16 (a 4-bit tag), got {slots}"
        # A queue of one entry would have a zero-width address in the generated RTL
        for name in ("ld_queue_length", "ex_queue_length", "st_queue_length"):
            assert getattr(self, name) >= 2, f"{name} must be at least 2"
        # Derived row geometry: rows must be whole bytes, and every bank must hold
        # at least one row (otherwise sp_width_bytes / bank depths are zero)
        assert self.sp_width % 8 == 0, (
            f"scratchpad row ({self.sp_width} bits) must be a whole number of bytes"
        )
        assert self.combiner_width % 8 == 0, (
            f"Combiner row ({self.combiner_width} bits) must be a whole number of bytes"
        )
        assert self.sp_bank_entries >= 1, (
            f"scratchpad of {self.sp_capacity_kb} KiB is too small for {self.sp_banks} "
            f"banks of {self.sp_width}-bit rows"
        )
        assert self.combiner_bank_entries >= 1, (
            f"Combiner of {self.combiner_capacity_kb} KiB is too small for {self.combiner_banks} "
            f"banks of {self.combiner_width}-bit rows"
        )


# Pre-defined configurations
DEFAULT_CONFIG = SystolicConfig()
"""Default configuration."""

LEAN_CONFIG = SystolicConfig(dataflow=Dataflow.B_STATIONARY)
"""Lean configuration for faster simulation."""

CHIP_CONFIG = SystolicConfig(
    sp_capacity_kb=64,
    combiner_capacity_kb=32,
    dataflow=Dataflow.B_STATIONARY,
)
"""Configuration optimized for tapeout."""

LARGE_CONFIG = SystolicConfig(
    grid_rows=32,
    grid_cols=32,
    sp_capacity_kb=512,
    combiner_capacity_kb=128,
)
"""Large configuration for high-performance applications."""
