"""
Stencil Machine Configuration Module

This module defines the configuration dataclass for the stencil machine.
All hardware parameters for line buffers, window former, and MAC array
are specified here and propagate through the design.
"""

from dataclasses import dataclass
from enum import Enum


class StencilActivation(Enum):
    """Supported activation functions for stencil machine output."""

    NONE = 0
    RELU = 1
    RELU6 = 2
    CLIP = 3


class StencilDataflow(Enum):
    """Dataflow modes for the stencil machine."""

    OUTPUT_STATIONARY = 0  # Accumulate output in place (default)
    WEIGHT_STATIONARY = 1  # Keep filter weights in PE registers


@dataclass
class StencilConfig:
    """
    Configuration for the stencil machine.

    This dataclass contains all parameters needed to generate a complete
    stencil machine for Conv2D and spatial operations.

    Example:
        >>> config = StencilConfig(max_width=224, parallel_channels=32)
        >>> print(config.line_buffer_size_kb)
    """

    # =========================================================================
    # Input Constraints
    # =========================================================================
    max_width: int = 224
    """Maximum supported input width (determines line buffer depth)."""

    max_height: int = 224
    """Maximum supported input height."""

    max_in_channels: int = 2048
    """Maximum supported input channels."""

    max_out_channels: int = 2048
    """Maximum supported output channels."""

    max_batch: int = 16
    """Maximum batch size."""

    # =========================================================================
    # Kernel Constraints
    # =========================================================================
    max_kernel_h: int = 7
    """Maximum kernel height (typically 7 for largest common kernels)."""

    max_kernel_w: int = 7
    """Maximum kernel width (typically 7 for largest common kernels)."""

    max_stride: int = 4
    """Maximum stride (1, 2, or 4 typical)."""

    max_dilation: int = 2
    """Maximum dilation factor."""

    # =========================================================================
    # Data Types (bit widths)
    # =========================================================================
    input_bits: int = 8
    """Bit width of input activations (typically 8 for INT8)."""

    weight_bits: int = 8
    """Bit width of filter weights (typically 8 for INT8)."""

    acc_bits: int = 32
    """Bit width of accumulator (typically 32 for INT32)."""

    output_bits: int = 8
    """Bit width of output after quantization (typically 8)."""

    # =========================================================================
    # Parallelism Configuration
    # =========================================================================
    parallel_channels: int = 32
    """Number of output channels computed in parallel (P_c)."""

    channel_serial: bool = True
    """
    If True, process input channels serially (area efficient).
    If False, process all input channels in parallel (higher throughput).
    """

    # =========================================================================
    # Line Buffer Configuration
    # =========================================================================
    line_buffer_banks: int = 1
    """Number of line buffer banks for double buffering rows."""

    # =========================================================================
    # Filter Buffer Configuration
    # =========================================================================
    filter_buffer_kb: int = 64
    """Filter buffer capacity in KB (for caching filter tiles)."""

    filter_double_buffer: bool = True
    """Enable double buffering for filter loads."""

    # =========================================================================
    # Output Configuration
    # =========================================================================
    output_buffer_depth: int = 4
    """Depth of output buffer (in output rows)."""

    # =========================================================================
    # Pipeline Configuration
    # =========================================================================
    mac_pipeline_stages: int = 1
    """Pipeline stages in MAC units (0 = combinational)."""

    adder_tree_pipeline: bool = False
    """Pipeline the adder tree (for high frequencies)."""

    # =========================================================================
    # Feature Flags
    # =========================================================================
    has_depthwise: bool = True
    """Enable depthwise convolution support."""

    has_pooling: bool = True
    """Enable max/avg pooling support."""

    has_activation: bool = True
    """Enable activation functions (ReLU, etc.)."""

    has_quantization: bool = True
    """Enable output quantization."""

    has_bias: bool = True
    """Enable bias addition."""

    # =========================================================================
    # Dataflow
    # =========================================================================
    dataflow: StencilDataflow = StencilDataflow.OUTPUT_STATIONARY
    """Default dataflow mode."""

    # =========================================================================
    # Computed Properties
    # =========================================================================
    @property
    def line_buffer_width(self) -> int:
        """Width of each line buffer in bits."""
        return self.max_width * self.input_bits

    @property
    def line_buffer_width_bytes(self) -> int:
        """Width of each line buffer in bytes."""
        return self.line_buffer_width // 8

    @property
    def line_buffer_depth(self) -> int:
        """Number of line buffers needed (= max_kernel_h)."""
        return self.max_kernel_h

    @property
    def line_buffer_total_bits(self) -> int:
        """Total line buffer storage in bits (per channel)."""
        return self.line_buffer_depth * self.line_buffer_width

    @property
    def line_buffer_size_kb(self) -> float:
        """
        Line buffer size in KB.

        Note: This is per input channel. For channel-serial mode,
        only one channel's worth of line buffers is needed.
        For channel-parallel, multiply by max_in_channels.
        """
        return self.line_buffer_total_bits / 8 / 1024

    @property
    def window_size(self) -> int:
        """Maximum window size in elements."""
        return self.max_kernel_h * self.max_kernel_w

    @property
    def window_bits(self) -> int:
        """Window size in bits (for one input channel)."""
        return self.window_size * self.input_bits

    @property
    def macs_per_bank(self) -> int:
        """Number of MAC operations per MAC bank per cycle."""
        return self.max_kernel_h * self.max_kernel_w

    @property
    def total_macs(self) -> int:
        """Total MAC units in the array."""
        return self.parallel_channels * self.macs_per_bank

    @property
    def filter_buffer_entries(self) -> int:
        """Number of filter coefficient sets that fit in buffer."""
        filter_size = self.max_kernel_h * self.max_kernel_w * self.weight_bits
        total_bits = self.filter_buffer_kb * 1024 * 8
        # Each output channel needs filter_size bits per input channel
        return total_bits // (filter_size * self.parallel_channels)

    @property
    def peak_throughput_gops(self) -> float:
        """Peak throughput in GOPS at 1 GHz (for INT8×INT8)."""
        # One window computation per cycle when fully pipelined
        return self.total_macs * 2 / 1e9  # 2 ops per MAC (mult + add)

    @property
    def multiplier_width(self) -> int:
        """Output width of each multiplier."""
        return self.input_bits + self.weight_bits

    @property
    def adder_tree_depth(self) -> int:
        """Depth of adder tree for reducing MAC products."""
        import math

        return math.ceil(math.log2(self.macs_per_bank)) if self.macs_per_bank > 1 else 0

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.max_width > 0, "max_width must be positive"
        assert self.max_height > 0, "max_height must be positive"
        assert self.max_in_channels > 0, "max_in_channels must be positive"
        assert self.max_out_channels > 0, "max_out_channels must be positive"
        assert 1 <= self.max_kernel_h <= 7, "max_kernel_h must be 1-7"
        assert 1 <= self.max_kernel_w <= 7, "max_kernel_w must be 1-7"
        assert self.input_bits > 0, "input_bits must be positive"
        assert self.weight_bits > 0, "weight_bits must be positive"
        assert self.acc_bits >= self.input_bits + self.weight_bits, (
            "acc_bits should be >= input_bits + weight_bits"
        )
        assert self.parallel_channels > 0, "parallel_channels must be positive"
        assert self.parallel_channels & (self.parallel_channels - 1) == 0, (
            "parallel_channels should be a power of 2"
        )


# Pre-defined configurations
DEFAULT_STENCIL_CONFIG = StencilConfig()
"""Default stencil machine configuration."""

SMALL_STENCIL_CONFIG = StencilConfig(
    max_width=56,
    max_height=56,
    max_in_channels=512,
    max_out_channels=512,
    parallel_channels=16,
    filter_buffer_kb=16,
)
"""Small configuration for testing and edge devices."""

LARGE_STENCIL_CONFIG = StencilConfig(
    max_width=224,
    max_height=224,
    max_in_channels=2048,
    max_out_channels=2048,
    parallel_channels=64,
    filter_buffer_kb=128,
)
"""Large configuration for high-performance inference."""

EDGE_STENCIL_CONFIG = StencilConfig(
    max_width=112,
    max_height=112,
    max_in_channels=256,
    max_out_channels=256,
    parallel_channels=16,
    filter_buffer_kb=8,
    mac_pipeline_stages=0,  # Combinational for lower latency
)
"""Edge-optimized configuration for low-power devices."""
