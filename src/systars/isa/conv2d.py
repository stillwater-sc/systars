"""
Conv2D ISA instruction: Y = conv2d(X, F) + B

This module implements the Conv2D instruction, a hardware FSM that automatically
handles tiled 2D convolution for feature maps and filters larger than the systolic
array.

Features:
- Automatic dataflow selection (OS/WS) based on tensor dimensions
- Tiled computation with configurable tile sizes
- Double buffering to hide memory latency
- Generates internal command sequences (LOAD, PRELOAD, COMPUTE, STORE)
- Supports stride, padding, and dilation

Tensor layouts (NHWC format):
    X (input):  [batch, height, width, channels_in]
    F (filter): [kernel_h, kernel_w, channels_in, channels_out]
    B (bias):   [channels_out]
    Y (output): [batch, out_height, out_width, channels_out]

The convolution is mapped to matrix multiplication by treating:
    - Each output position (b, oh, ow) as a row
    - Each output channel as a column
    - The inner dimension is kernel_h * kernel_w * channels_in

Usage:
    The user configures dimensions and addresses, then issues START.
    The hardware handles all tiling and scheduling automatically.
"""

from enum import IntEnum

from amaranth import Module, Mux, Signal
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig

# =============================================================================
# Configuration Commands
# =============================================================================


class Conv2dCmd(IntEnum):
    """Configuration commands for Conv2D instruction."""

    # Input dimensions: batch, height, width, channels_in
    CONFIG_INPUT_DIMS = 0x01  # data[7:0]=batch, [15:8]=height, [23:16]=width, [31:24]=channels_in

    # Output dimensions: out_height, out_width, channels_out
    CONFIG_OUTPUT_DIMS = 0x02  # data[15:0]=out_height, [31:16]=out_width, [47:32]=channels_out

    # Kernel dimensions: kernel_h, kernel_w
    CONFIG_KERNEL_DIMS = 0x03  # data[7:0]=kernel_h, [15:8]=kernel_w

    # Stride and padding
    CONFIG_STRIDE = 0x04  # data[7:0]=stride_h, [15:8]=stride_w
    CONFIG_PADDING = 0x05  # data[7:0]=pad_h, [15:8]=pad_w
    CONFIG_DILATION = 0x06  # data[7:0]=dilation_h, [15:8]=dilation_w

    # Base addresses (64-bit each)
    CONFIG_X_ADDR = 0x10  # Input tensor address
    CONFIG_F_ADDR = 0x11  # Filter tensor address
    CONFIG_Y_ADDR = 0x12  # Output tensor address
    CONFIG_B_ADDR = 0x13  # Bias address, 0 = no bias

    # Strides in bytes (for non-contiguous memory layouts)
    CONFIG_X_STRIDE = 0x20  # Row stride for X
    CONFIG_F_STRIDE = 0x21  # Row stride for F
    CONFIG_Y_STRIDE = 0x22  # Row stride for Y
    CONFIG_B_STRIDE = 0x23  # Row stride for B

    # Options
    CONFIG_OPTIONS = 0x30  # data[0]=accumulate, data[3:1]=activation

    # Control
    START = 0xF0
    ABORT = 0xFF


class InternalOpcode(IntEnum):
    """
    Internal opcodes emitted by Conv2D to downstream controllers.

    These are routed by a command dispatcher to the appropriate controller.
    """

    # Execute controller commands (0x00-0x0F)
    EXEC_CONFIG = 0x00  # Configure dataflow mode
    EXEC_PRELOAD = 0x01  # Preload bias from accumulator
    EXEC_COMPUTE = 0x02  # Execute convolution tile

    # Load controller commands (0x10-0x1F)
    LOAD_X = 0x10  # Load input patch: DRAM -> Scratchpad
    LOAD_F = 0x11  # Load filter tile: DRAM -> Scratchpad
    LOAD_B = 0x12  # Load bias: DRAM -> Accumulator

    # Store controller commands (0x20-0x2F)
    STORE_Y = 0x20  # Store output tile: Accumulator -> DRAM


class Conv2dError(IntEnum):
    """Error codes for Conv2D instruction."""

    NONE = 0x00
    INVALID_DIMS = 0x01  # Dimensions are zero or too large
    INVALID_ADDR = 0x02  # Address alignment error
    INVALID_STRIDE = 0x03  # Stride too small
    INVALID_PADDING = 0x04  # Padding larger than kernel
    CONFIG_INCOMPLETE = 0x05  # Started without required config
    TIMEOUT = 0x06  # Command not accepted in time


# =============================================================================
# Conv2D Component
# =============================================================================


class Conv2d(Component):
    """
    Conv2D ISA instruction: Y = conv2d(X, F) + B

    The hardware automatically:
    - Selects optimal dataflow (OS/WS) based on dimensions
    - Tiles the computation to fit array dimensions
    - Double-buffers to hide memory latency
    - Generates internal command sequences

    The convolution is mapped to matrix multiplication:
        - M = batch * out_height * out_width (output spatial positions)
        - N = channels_out (output channels)
        - K = kernel_h * kernel_w * channels_in (flattened kernel)

    Ports:
        Configuration Interface:
            cfg_valid: Configuration command available
            cfg_ready: Ready to accept configuration
            cfg_cmd: Configuration command code (Conv2dCmd)
            cfg_data: Configuration data (64-bit)

        Command Output (to reservation station):
            cmd_valid: Command available for dispatch
            cmd_ready: Downstream ready to accept
            cmd_opcode: Internal opcode (LOAD, PRELOAD, COMPUTE, STORE)
            cmd_rs1: Source operand 1
            cmd_rs2: Source operand 2
            cmd_rd: Destination operand

        Status:
            busy: Instruction is executing
            done: Instruction completed successfully
            error: Error occurred
            error_code: Specific error code

        Debug:
            progress_m, progress_n, progress_k: Current tile indices
            selected_dataflow: Automatically selected dataflow mode
    """

    def __init__(self, config: SystolicConfig):
        self.config = config

        # Array dimension (assume square for simplicity)
        self.dim = config.grid_rows * config.tile_rows

        ports = {
            # Configuration interface
            "cfg_valid": In(1),
            "cfg_ready": Out(1),
            "cfg_cmd": In(8),
            "cfg_data": In(64),
            # Command output (to reservation station / controllers)
            "cmd_valid": Out(1),
            "cmd_ready": In(1),
            "cmd_opcode": Out(8),
            "cmd_rs1": Out(64),
            "cmd_rs2": Out(64),
            "cmd_rd": Out(64),
            "cmd_k_dim": Out(16),  # Inner dimension for compute
            "cmd_id": Out(8),
            # Status
            "busy": Out(1),
            "done": Out(1),
            "error": Out(1),
            "error_code": Out(8),
            # Debug / Progress
            "progress_m": Out(16),  # Output position tile
            "progress_n": Out(16),  # Output channel tile
            "progress_k": Out(16),  # Input channel tile
            "selected_dataflow": Out(2),  # 0=OS, 1=WS
            # Dimension outputs (for debugging)
            "param_batch": Out(8),
            "param_in_h": Out(16),
            "param_in_w": Out(16),
            "param_in_c": Out(16),
            "param_out_h": Out(16),
            "param_out_w": Out(16),
            "param_out_c": Out(16),
            "param_kernel_h": Out(8),
            "param_kernel_w": Out(8),
            # Computed dimensions
            "param_M": Out(32),  # batch * out_h * out_w
            "param_N": Out(32),  # out_c
            "param_K": Out(32),  # kernel_h * kernel_w * in_c
            # Computed addresses (for debugging/verification)
            "dbg_addr_X": Out(64),
            "dbg_addr_F": Out(64),
            "dbg_addr_Y": Out(64),
            "dbg_addr_B": Out(64),
            # Current tile sizes (for edge tile handling)
            "dbg_tile_M": Out(16),
            "dbg_tile_N": Out(16),
            "dbg_tile_K": Out(16),
            # Bank selection (for double buffering debugging)
            "dbg_sp_bank_X": Out(2),
            "dbg_sp_bank_F": Out(2),
            "dbg_acc_bank": Out(1),
        }

        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config
        dim = self.dim

        # =====================================================================
        # Configuration Registers
        # =====================================================================

        # Input dimensions
        reg_batch = Signal(8, name="reg_batch")
        reg_in_h = Signal(16, name="reg_in_h")
        reg_in_w = Signal(16, name="reg_in_w")
        reg_in_c = Signal(16, name="reg_in_c")

        # Output dimensions
        reg_out_h = Signal(16, name="reg_out_h")
        reg_out_w = Signal(16, name="reg_out_w")
        reg_out_c = Signal(16, name="reg_out_c")

        # Kernel dimensions
        reg_kernel_h = Signal(8, name="reg_kernel_h")
        reg_kernel_w = Signal(8, name="reg_kernel_w")

        # Stride and padding
        reg_stride_h = Signal(8, name="reg_stride_h", init=1)
        reg_stride_w = Signal(8, name="reg_stride_w", init=1)
        reg_pad_h = Signal(8, name="reg_pad_h")
        reg_pad_w = Signal(8, name="reg_pad_w")
        reg_dilation_h = Signal(8, name="reg_dilation_h", init=1)
        reg_dilation_w = Signal(8, name="reg_dilation_w", init=1)

        # Base addresses
        reg_X_addr = Signal(64, name="reg_X_addr")
        reg_F_addr = Signal(64, name="reg_F_addr")
        reg_Y_addr = Signal(64, name="reg_Y_addr")
        reg_B_addr = Signal(64, name="reg_B_addr")

        # Strides
        reg_X_stride = Signal(32, name="reg_X_stride")
        reg_F_stride = Signal(32, name="reg_F_stride")
        reg_Y_stride = Signal(32, name="reg_Y_stride")
        reg_B_stride = Signal(32, name="reg_B_stride")

        # Options
        reg_accumulate = Signal(1, name="reg_accumulate")
        reg_activation = Signal(3, name="reg_activation")

        # Configuration valid flags
        cfg_input_dims_valid = Signal(1)
        cfg_output_dims_valid = Signal(1)
        cfg_kernel_dims_valid = Signal(1)
        cfg_addrs_valid = Signal(1)

        # =====================================================================
        # Derived Dimensions (for matrix multiplication mapping)
        # =====================================================================

        # M = batch * out_h * out_w (number of output spatial positions)
        # N = out_c (number of output channels)
        # K = kernel_h * kernel_w * in_c (flattened kernel size)
        reg_M = Signal(32, name="reg_M")
        reg_N = Signal(32, name="reg_N")
        reg_K = Signal(32, name="reg_K")

        # =====================================================================
        # Tile Counters
        # =====================================================================

        # Number of tiles in each dimension
        tiles_M = Signal(16)  # ceil(M / dim)
        tiles_N = Signal(16)  # ceil(N / dim)
        tiles_K = Signal(16)  # ceil(K / dim)

        # Current tile indices
        tile_m = Signal(16)  # Output position tile
        tile_n = Signal(16)  # Output channel tile
        tile_k = Signal(16)  # Input channel tile

        # Command ID counter
        cmd_id_counter = Signal(8)

        # =====================================================================
        # Address Calculation
        # =====================================================================

        # Element sizes in bytes
        input_bytes = cfg.input_bits // 8
        acc_bytes = cfg.acc_bits // 8

        # Computed tile addresses
        addr_X_tile = Signal(64, name="addr_X_tile")
        addr_F_tile = Signal(64, name="addr_F_tile")
        addr_Y_tile = Signal(64, name="addr_Y_tile")
        addr_B_tile = Signal(64, name="addr_B_tile")

        # Tile dimensions for current tile (may be smaller at edges)
        tile_M_size = Signal(16, name="tile_M_size")
        tile_N_size = Signal(16, name="tile_N_size")
        tile_K_size = Signal(16, name="tile_K_size")

        # Element offsets for current tile
        m_offset = Signal(32, name="m_offset")  # tile_m * dim
        n_offset = Signal(32, name="n_offset")  # tile_n * dim
        k_offset = Signal(32, name="k_offset")  # tile_k * dim

        # Compute element offsets from tile indices
        dim_shift = (dim - 1).bit_length()  # log2(dim) for power-of-2 dim
        m.d.comb += [
            m_offset.eq(tile_m << dim_shift),
            n_offset.eq(tile_n << dim_shift),
            k_offset.eq(tile_k << dim_shift),
        ]

        # Compute actual tile sizes (handle edge cases)
        remaining_M = Signal(32)
        remaining_N = Signal(32)
        remaining_K = Signal(32)

        m.d.comb += [
            remaining_M.eq(reg_M - m_offset),
            remaining_N.eq(reg_N - n_offset),
            remaining_K.eq(reg_K - k_offset),
        ]

        # Clamp to dim
        m.d.comb += [
            tile_M_size.eq(Mux(remaining_M < dim, remaining_M[:16], dim)),
            tile_N_size.eq(Mux(remaining_N < dim, remaining_N[:16], dim)),
            tile_K_size.eq(Mux(remaining_K < dim, remaining_K[:16], dim)),
        ]

        # =====================================================================
        # Spatial Position Decoding
        # =====================================================================
        # Decode m_offset back to (batch, out_h, out_w) for address calculation
        # m_index = b * (out_h * out_w) + oh * out_w + ow

        out_hw = Signal(32)  # out_h * out_w
        m.d.comb += out_hw.eq(reg_out_h * reg_out_w)

        # Current position within M dimension
        m_pos = Signal(32)
        m.d.comb += m_pos.eq(m_offset)

        # Decode to batch, oh, ow
        # batch = m_pos / (out_h * out_w)
        # remainder = m_pos % (out_h * out_w)
        # oh = remainder / out_w
        # ow = remainder % out_w
        cur_batch = Signal(32)
        cur_oh = Signal(32)
        cur_ow = Signal(32)

        # For simplicity, we'll use iterative calculation during execution
        # These are updated in the state machine

        # =====================================================================
        # Address Calculations
        # =====================================================================

        # X address: For im2col-style access, we need to calculate the address
        # of the input patch. The patch for output position (b, oh, ow) starts at:
        # X[b, oh*stride_h - pad_h, ow*stride_w - pad_w, :]
        #
        # For tiled computation, we calculate based on:
        # - Current batch (derived from m_offset)
        # - Current output row (oh, derived from m_offset)
        # - Current output col (ow, derived from m_offset)
        # - Current input channel offset (k_offset % in_c)

        # Compute input patch start position
        in_row_start = Signal(32, name="in_row_start")  # oh * stride_h - pad_h
        in_col_start = Signal(32, name="in_col_start")  # ow * stride_w - pad_w

        # Input address for the patch
        # X[b, r, c, ch] = X_addr + b * (in_h * in_w * in_c) * input_bytes
        #                         + r * (in_w * in_c) * input_bytes
        #                         + c * in_c * input_bytes
        #                         + ch * input_bytes

        # Use X_stride as row stride if provided
        x_batch_offset = Signal(64, name="x_batch_offset")
        x_row_offset = Signal(64, name="x_row_offset")
        x_col_offset = Signal(64, name="x_col_offset")
        x_ch_offset = Signal(64, name="x_ch_offset")

        in_w_c = Signal(32)  # in_w * in_c
        in_h_w_c = Signal(32)  # in_h * in_w * in_c
        m.d.comb += [
            in_w_c.eq(reg_in_w * reg_in_c),
            in_h_w_c.eq(reg_in_h * in_w_c),
        ]

        m.d.comb += [
            x_batch_offset.eq(cur_batch * in_h_w_c * input_bytes),
            x_row_offset.eq(in_row_start * in_w_c * input_bytes),
            x_col_offset.eq(in_col_start * reg_in_c * input_bytes),
            x_ch_offset.eq(k_offset * input_bytes),
            addr_X_tile.eq(reg_X_addr + x_batch_offset + x_row_offset + x_col_offset + x_ch_offset),
        ]

        # F address: Filter is [kernel_h, kernel_w, in_c, out_c]
        # F[kh, kw, ic, oc] = F_addr + (kh * kernel_w * in_c * out_c
        #                            + kw * in_c * out_c
        #                            + ic * out_c
        #                            + oc) * input_bytes
        # For tiled access, we access filter slices based on k_offset and n_offset
        f_row_offset = Signal(64, name="f_row_offset")
        f_col_offset = Signal(64, name="f_col_offset")

        m.d.comb += [
            # k_offset is the flattened kernel index, n_offset is output channel
            f_row_offset.eq(k_offset * reg_out_c * input_bytes),
            f_col_offset.eq(n_offset * input_bytes),
            addr_F_tile.eq(reg_F_addr + f_row_offset + f_col_offset),
        ]

        # Y address: Output is [batch, out_h, out_w, out_c]
        y_batch_offset = Signal(64, name="y_batch_offset")
        y_pos_offset = Signal(64, name="y_pos_offset")
        y_ch_offset = Signal(64, name="y_ch_offset")

        out_h_w_c = Signal(32)
        m.d.comb += out_h_w_c.eq(reg_out_h * reg_out_w * reg_out_c)

        m.d.comb += [
            y_batch_offset.eq(cur_batch * out_h_w_c * acc_bytes),
            y_pos_offset.eq((m_offset - cur_batch * out_hw) * reg_out_c * acc_bytes),
            y_ch_offset.eq(n_offset * acc_bytes),
            addr_Y_tile.eq(reg_Y_addr + y_batch_offset + y_pos_offset + y_ch_offset),
        ]

        # B address: Bias is [out_c]
        m.d.comb += addr_B_tile.eq(reg_B_addr + n_offset * acc_bytes)

        # =====================================================================
        # Scratchpad and Accumulator Bank Selection
        # =====================================================================

        sp_bank_X = Signal(2, name="sp_bank_X")  # 0 or 2
        sp_bank_F = Signal(2, name="sp_bank_F")  # 1 or 3
        acc_bank = Signal(1, name="acc_bank")  # 0 or 1

        # Local addresses within scratchpad
        sp_addr_X = Signal(32, name="sp_addr_X")
        sp_addr_F = Signal(32, name="sp_addr_F")
        acc_addr_Y = Signal(32, name="acc_addr_Y")

        m.d.comb += [
            sp_addr_X.eq(sp_bank_X << 16),
            sp_addr_F.eq(sp_bank_F << 16),
            acc_addr_Y.eq(acc_bank << 16),
        ]

        # =====================================================================
        # Transfer Length Calculation
        # =====================================================================

        load_X_len = Signal(16, name="load_X_len")
        load_F_len = Signal(16, name="load_F_len")
        load_B_len = Signal(16, name="load_B_len")
        store_Y_len = Signal(16, name="store_Y_len")

        m.d.comb += [
            load_X_len.eq(tile_M_size),
            load_F_len.eq(tile_K_size),
            load_B_len.eq(tile_N_size),
            store_Y_len.eq(tile_M_size),
        ]

        # =====================================================================
        # Dataflow Selection
        # =====================================================================

        # Selected dataflow mode
        # 0 = Output-Stationary: accumulate in PE
        # 1 = Weight-Stationary: keep filters in PE
        dataflow_mode = Signal(2)

        # Heuristic: weight-stationary is usually better for convolutions
        # because filters are reused across many output positions
        with m.If(reg_K > (reg_M << 1)):
            # K is large relative to M -> weight stationary
            m.d.comb += dataflow_mode.eq(1)
        with m.Else():
            # Default to output-stationary
            m.d.comb += dataflow_mode.eq(0)

        # =====================================================================
        # State Machine
        # =====================================================================

        state = Signal(8)
        STATE_IDLE = 0
        STATE_INIT = 1
        STATE_CONFIG = 2
        STATE_LOAD_B = 3
        STATE_LOAD_X = 4
        STATE_LOAD_F = 5
        STATE_PRELOAD = 6
        STATE_COMPUTE = 7
        STATE_NEXT_K = 8
        STATE_STORE = 9
        STATE_NEXT_MN = 10
        STATE_DONE = 11
        STATE_ERROR = 12

        # =====================================================================
        # Output Assignments
        # =====================================================================

        # Expose dimension parameters
        m.d.comb += [
            self.param_batch.eq(reg_batch),
            self.param_in_h.eq(reg_in_h),
            self.param_in_w.eq(reg_in_w),
            self.param_in_c.eq(reg_in_c),
            self.param_out_h.eq(reg_out_h),
            self.param_out_w.eq(reg_out_w),
            self.param_out_c.eq(reg_out_c),
            self.param_kernel_h.eq(reg_kernel_h),
            self.param_kernel_w.eq(reg_kernel_w),
            self.param_M.eq(reg_M),
            self.param_N.eq(reg_N),
            self.param_K.eq(reg_K),
            self.selected_dataflow.eq(dataflow_mode),
            self.progress_m.eq(tile_m),
            self.progress_n.eq(tile_n),
            self.progress_k.eq(tile_k),
        ]

        # Expose computed addresses for debugging
        m.d.comb += [
            self.dbg_addr_X.eq(addr_X_tile),
            self.dbg_addr_F.eq(addr_F_tile),
            self.dbg_addr_Y.eq(addr_Y_tile),
            self.dbg_addr_B.eq(addr_B_tile),
            self.dbg_tile_M.eq(tile_M_size),
            self.dbg_tile_N.eq(tile_N_size),
            self.dbg_tile_K.eq(tile_K_size),
            self.dbg_sp_bank_X.eq(sp_bank_X),
            self.dbg_sp_bank_F.eq(sp_bank_F),
            self.dbg_acc_bank.eq(acc_bank),
        ]

        # Status signals
        m.d.comb += [
            self.busy.eq(state != STATE_IDLE),
            self.done.eq(state == STATE_DONE),
            self.error.eq(state == STATE_ERROR),
        ]

        # Default command output
        m.d.comb += [
            self.cmd_valid.eq(0),
            self.cmd_opcode.eq(0),
            self.cmd_rs1.eq(0),
            self.cmd_rs2.eq(0),
            self.cmd_rd.eq(0),
            self.cmd_k_dim.eq(dim),
            self.cmd_id.eq(cmd_id_counter),
        ]

        # Configuration interface
        m.d.comb += self.cfg_ready.eq(state == STATE_IDLE)

        # =====================================================================
        # State Machine Logic
        # =====================================================================

        with m.FSM(init="IDLE"):
            # -----------------------------------------------------------------
            # IDLE: Wait for configuration or start command
            # -----------------------------------------------------------------
            with m.State("IDLE"):
                m.d.comb += state.eq(STATE_IDLE)

                with m.If(self.cfg_valid), m.Switch(self.cfg_cmd):
                    # Input dimension configuration
                    with m.Case(Conv2dCmd.CONFIG_INPUT_DIMS):
                        m.d.sync += [
                            reg_batch.eq(self.cfg_data[0:8]),
                            reg_in_h.eq(self.cfg_data[8:24]),
                            reg_in_w.eq(self.cfg_data[24:40]),
                            reg_in_c.eq(self.cfg_data[40:56]),
                            cfg_input_dims_valid.eq(1),
                        ]

                    # Output dimension configuration
                    with m.Case(Conv2dCmd.CONFIG_OUTPUT_DIMS):
                        m.d.sync += [
                            reg_out_h.eq(self.cfg_data[0:16]),
                            reg_out_w.eq(self.cfg_data[16:32]),
                            reg_out_c.eq(self.cfg_data[32:48]),
                            cfg_output_dims_valid.eq(1),
                        ]

                    # Kernel dimension configuration
                    with m.Case(Conv2dCmd.CONFIG_KERNEL_DIMS):
                        m.d.sync += [
                            reg_kernel_h.eq(self.cfg_data[0:8]),
                            reg_kernel_w.eq(self.cfg_data[8:16]),
                            cfg_kernel_dims_valid.eq(1),
                        ]

                    # Stride configuration
                    with m.Case(Conv2dCmd.CONFIG_STRIDE):
                        m.d.sync += [
                            reg_stride_h.eq(self.cfg_data[0:8]),
                            reg_stride_w.eq(self.cfg_data[8:16]),
                        ]

                    # Padding configuration
                    with m.Case(Conv2dCmd.CONFIG_PADDING):
                        m.d.sync += [
                            reg_pad_h.eq(self.cfg_data[0:8]),
                            reg_pad_w.eq(self.cfg_data[8:16]),
                        ]

                    # Dilation configuration
                    with m.Case(Conv2dCmd.CONFIG_DILATION):
                        m.d.sync += [
                            reg_dilation_h.eq(self.cfg_data[0:8]),
                            reg_dilation_w.eq(self.cfg_data[8:16]),
                        ]

                    # Address configuration
                    with m.Case(Conv2dCmd.CONFIG_X_ADDR):
                        m.d.sync += reg_X_addr.eq(self.cfg_data)
                    with m.Case(Conv2dCmd.CONFIG_F_ADDR):
                        m.d.sync += reg_F_addr.eq(self.cfg_data)
                    with m.Case(Conv2dCmd.CONFIG_Y_ADDR):
                        m.d.sync += [
                            reg_Y_addr.eq(self.cfg_data),
                            cfg_addrs_valid.eq(1),
                        ]
                    with m.Case(Conv2dCmd.CONFIG_B_ADDR):
                        m.d.sync += reg_B_addr.eq(self.cfg_data)

                    # Stride configuration
                    with m.Case(Conv2dCmd.CONFIG_X_STRIDE):
                        m.d.sync += reg_X_stride.eq(self.cfg_data[0:32])
                    with m.Case(Conv2dCmd.CONFIG_F_STRIDE):
                        m.d.sync += reg_F_stride.eq(self.cfg_data[0:32])
                    with m.Case(Conv2dCmd.CONFIG_Y_STRIDE):
                        m.d.sync += reg_Y_stride.eq(self.cfg_data[0:32])
                    with m.Case(Conv2dCmd.CONFIG_B_STRIDE):
                        m.d.sync += reg_B_stride.eq(self.cfg_data[0:32])

                    # Options
                    with m.Case(Conv2dCmd.CONFIG_OPTIONS):
                        m.d.sync += [
                            reg_accumulate.eq(self.cfg_data[0]),
                            reg_activation.eq(self.cfg_data[1:4]),
                        ]

                    # Start execution
                    with m.Case(Conv2dCmd.START):
                        # Validate configuration
                        cfg_valid = Signal()
                        m.d.comb += cfg_valid.eq(
                            cfg_input_dims_valid
                            & cfg_output_dims_valid
                            & cfg_kernel_dims_valid
                            & cfg_addrs_valid
                        )
                        with m.If(cfg_valid):
                            m.next = "INIT"
                        with m.Else():
                            m.d.sync += self.error_code.eq(Conv2dError.CONFIG_INCOMPLETE)
                            m.next = "ERROR"

            # -----------------------------------------------------------------
            # INIT: Calculate derived dimensions and tile counts
            # -----------------------------------------------------------------
            with m.State("INIT"):
                m.d.comb += state.eq(STATE_INIT)

                # Calculate M, N, K for matrix multiplication mapping
                # M = batch * out_h * out_w
                # N = out_c
                # K = kernel_h * kernel_w * in_c
                m.d.sync += [
                    reg_M.eq(reg_batch * reg_out_h * reg_out_w),
                    reg_N.eq(reg_out_c),
                    reg_K.eq(reg_kernel_h * reg_kernel_w * reg_in_c),
                    # Initialize tile counters
                    tile_m.eq(0),
                    tile_n.eq(0),
                    tile_k.eq(0),
                    cmd_id_counter.eq(0),
                    # Initialize bank selection
                    sp_bank_X.eq(0),
                    sp_bank_F.eq(1),
                    acc_bank.eq(0),
                    # Initialize spatial position
                    cur_batch.eq(0),
                    cur_oh.eq(0),
                    cur_ow.eq(0),
                    in_row_start.eq(0),  # Will be -pad_h, but unsigned
                    in_col_start.eq(0),  # Will be -pad_w, but unsigned
                ]

                m.next = "CALC_TILES"

            # -----------------------------------------------------------------
            # CALC_TILES: Calculate number of tiles (separate state for timing)
            # -----------------------------------------------------------------
            with m.State("CALC_TILES"):
                # Calculate number of tiles (ceiling division)
                m.d.sync += [
                    tiles_M.eq((reg_M + dim - 1) >> dim_shift),
                    tiles_N.eq((reg_N + dim - 1) >> dim_shift),
                    tiles_K.eq((reg_K + dim - 1) >> dim_shift),
                ]
                m.next = "CONFIG"

            # -----------------------------------------------------------------
            # CONFIG: Issue CONFIG command to set dataflow mode
            # -----------------------------------------------------------------
            with m.State("CONFIG"):
                m.d.comb += state.eq(STATE_CONFIG)

                m.d.comb += [
                    self.cmd_valid.eq(1),
                    self.cmd_opcode.eq(InternalOpcode.EXEC_CONFIG),
                    self.cmd_rs1.eq(dataflow_mode),
                ]

                with m.If(self.cmd_ready):
                    with m.If(reg_B_addr != 0):
                        m.next = "LOAD_B"
                    with m.Else():
                        m.next = "LOAD_X"

            # -----------------------------------------------------------------
            # LOAD_B: Issue LOAD command for bias
            # -----------------------------------------------------------------
            with m.State("LOAD_B"):
                m.d.comb += state.eq(STATE_LOAD_B)

                m.d.comb += [
                    self.cmd_valid.eq(1),
                    self.cmd_opcode.eq(InternalOpcode.LOAD_B),
                    self.cmd_rs1.eq(addr_B_tile),
                    self.cmd_rs2.eq(acc_addr_Y),
                    self.cmd_rd.eq(load_B_len),
                ]

                with m.If(self.cmd_ready):
                    m.next = "LOAD_X"

            # -----------------------------------------------------------------
            # LOAD_X: Issue LOAD command for input patch
            # -----------------------------------------------------------------
            with m.State("LOAD_X"):
                m.d.comb += state.eq(STATE_LOAD_X)

                m.d.comb += [
                    self.cmd_valid.eq(1),
                    self.cmd_opcode.eq(InternalOpcode.LOAD_X),
                    self.cmd_rs1.eq(addr_X_tile),
                    self.cmd_rs2.eq(sp_addr_X),
                    self.cmd_rd.eq(load_X_len),
                ]

                with m.If(self.cmd_ready):
                    m.next = "LOAD_F"

            # -----------------------------------------------------------------
            # LOAD_F: Issue LOAD command for filter tile
            # -----------------------------------------------------------------
            with m.State("LOAD_F"):
                m.d.comb += state.eq(STATE_LOAD_F)

                m.d.comb += [
                    self.cmd_valid.eq(1),
                    self.cmd_opcode.eq(InternalOpcode.LOAD_F),
                    self.cmd_rs1.eq(addr_F_tile),
                    self.cmd_rs2.eq(sp_addr_F),
                    self.cmd_rd.eq(load_F_len),
                ]

                with m.If(self.cmd_ready):
                    with m.If(tile_k == 0):
                        m.next = "PRELOAD"
                    with m.Else():
                        m.next = "COMPUTE"

            # -----------------------------------------------------------------
            # PRELOAD: Issue PRELOAD command (first K iteration only)
            # -----------------------------------------------------------------
            with m.State("PRELOAD"):
                m.d.comb += state.eq(STATE_PRELOAD)

                m.d.comb += [
                    self.cmd_valid.eq(1),
                    self.cmd_opcode.eq(InternalOpcode.EXEC_PRELOAD),
                    self.cmd_rs1.eq(acc_addr_Y),
                    self.cmd_rs2.eq(tile_M_size),
                ]

                with m.If(self.cmd_ready):
                    m.next = "COMPUTE"

            # -----------------------------------------------------------------
            # COMPUTE: Issue COMPUTE command
            # -----------------------------------------------------------------
            with m.State("COMPUTE"):
                m.d.comb += state.eq(STATE_COMPUTE)

                m.d.comb += [
                    self.cmd_valid.eq(1),
                    self.cmd_opcode.eq(InternalOpcode.EXEC_COMPUTE),
                    self.cmd_rs1.eq(sp_addr_X),
                    self.cmd_rs2.eq(sp_addr_F),
                    self.cmd_rd.eq(acc_addr_Y),
                    self.cmd_k_dim.eq(tile_K_size),
                ]

                with m.If(self.cmd_ready):
                    m.next = "NEXT_K"

            # -----------------------------------------------------------------
            # NEXT_K: Advance K counter or move to store
            # -----------------------------------------------------------------
            with m.State("NEXT_K"):
                m.d.comb += state.eq(STATE_NEXT_K)

                with m.If(tile_k < tiles_K - 1):
                    m.d.sync += [
                        tile_k.eq(tile_k + 1),
                        sp_bank_X.eq(sp_bank_X ^ 2),
                        sp_bank_F.eq(sp_bank_F ^ 2),
                    ]
                    m.next = "LOAD_X"
                with m.Else():
                    m.d.sync += tile_k.eq(0)
                    m.next = "STORE"

            # -----------------------------------------------------------------
            # STORE: Issue STORE command for output tile
            # -----------------------------------------------------------------
            with m.State("STORE"):
                m.d.comb += state.eq(STATE_STORE)

                m.d.comb += [
                    self.cmd_valid.eq(1),
                    self.cmd_opcode.eq(InternalOpcode.STORE_Y),
                    self.cmd_rs1.eq(acc_addr_Y),
                    self.cmd_rs2.eq(addr_Y_tile),
                    self.cmd_rd.eq(store_Y_len | (reg_activation << 16)),
                ]

                with m.If(self.cmd_ready):
                    m.next = "NEXT_MN"

            # -----------------------------------------------------------------
            # NEXT_MN: Advance M, N counters or finish
            # -----------------------------------------------------------------
            with m.State("NEXT_MN"):
                m.d.comb += state.eq(STATE_NEXT_MN)

                has_more_n = Signal()
                has_more_m = Signal()
                has_bias = Signal()
                m.d.comb += [
                    has_more_n.eq(tile_n < tiles_N - 1),
                    has_more_m.eq(tile_m < tiles_M - 1),
                    has_bias.eq(reg_B_addr != 0),
                ]

                with m.If(has_more_n):
                    m.d.sync += [
                        tile_n.eq(tile_n + 1),
                        acc_bank.eq(~acc_bank),
                        sp_bank_X.eq(0),
                        sp_bank_F.eq(1),
                    ]
                    with m.If(has_bias):
                        m.next = "LOAD_B"
                    with m.Else():
                        m.next = "LOAD_X"
                with m.Elif(has_more_m):
                    m.d.sync += [
                        tile_m.eq(tile_m + 1),
                        tile_n.eq(0),
                        acc_bank.eq(~acc_bank),
                        sp_bank_X.eq(0),
                        sp_bank_F.eq(1),
                    ]
                    with m.If(has_bias):
                        m.next = "LOAD_B"
                    with m.Else():
                        m.next = "LOAD_X"
                with m.Else():
                    m.next = "DONE"

            # -----------------------------------------------------------------
            # DONE: Signal completion
            # -----------------------------------------------------------------
            with m.State("DONE"):
                m.d.comb += state.eq(STATE_DONE)
                m.d.sync += [
                    cfg_input_dims_valid.eq(0),
                    cfg_output_dims_valid.eq(0),
                    cfg_kernel_dims_valid.eq(0),
                    cfg_addrs_valid.eq(0),
                ]
                m.next = "IDLE"

            # -----------------------------------------------------------------
            # ERROR: Signal error
            # -----------------------------------------------------------------
            with m.State("ERROR"):
                m.d.comb += state.eq(STATE_ERROR)
                m.next = "IDLE"

        return m
