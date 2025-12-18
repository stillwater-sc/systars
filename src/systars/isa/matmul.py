"""
Matmul ISA instruction: C = A @ B + D

This module implements the Matmul instruction, a hardware FSM that automatically
handles tiled matrix multiplication for matrices larger than the systolic array.

Features:
- Automatic dataflow selection (OS/AS/BS) based on tensor dimensions
- Tiled computation with configurable tile sizes
- Double buffering to hide memory latency
- Generates internal command sequences (LOAD, PRELOAD, COMPUTE, STORE)

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


class MatmulCmd(IntEnum):
    """Configuration commands for Matmul instruction."""

    # Matrix dimensions
    CONFIG_DIMS = 0x01  # data[15:0]=M, data[31:16]=N, data[47:32]=K

    # Base addresses (64-bit each)
    CONFIG_A_ADDR = 0x10
    CONFIG_B_ADDR = 0x11
    CONFIG_C_ADDR = 0x12
    CONFIG_D_ADDR = 0x13  # Bias address, 0 = no bias

    # Strides in bytes (32-bit each)
    CONFIG_A_STRIDE = 0x20
    CONFIG_B_STRIDE = 0x21
    CONFIG_C_STRIDE = 0x22
    CONFIG_D_STRIDE = 0x23

    # Options
    CONFIG_OPTIONS = 0x30  # data[0]=accumulate, data[3:1]=activation

    # Control
    START = 0xF0
    ABORT = 0xFF


class InternalOpcode(IntEnum):
    """
    Internal opcodes emitted by Matmul to downstream controllers.

    These are routed by a command dispatcher to the appropriate controller.
    """

    # Execute controller commands (0x00-0x0F)
    EXEC_CONFIG = 0x00  # Configure dataflow mode
    EXEC_PRELOAD = 0x01  # Preload bias from accumulator
    EXEC_COMPUTE = 0x02  # Execute matmul

    # Load controller commands (0x10-0x1F)
    LOAD_A = 0x10  # Load A tile: DRAM -> Scratchpad
    LOAD_B = 0x11  # Load B tile: DRAM -> Scratchpad
    LOAD_D = 0x12  # Load D tile: DRAM -> Accumulator (bias)

    # Store controller commands (0x20-0x2F)
    STORE_C = 0x20  # Store C tile: Accumulator -> DRAM


class MatmulError(IntEnum):
    """Error codes for Matmul instruction."""

    NONE = 0x00
    INVALID_DIMS = 0x01  # M, N, or K is zero or too large
    INVALID_ADDR = 0x02  # Address alignment error
    INVALID_STRIDE = 0x03  # Stride too small for row
    CONFIG_INCOMPLETE = 0x04  # Started without required config
    TIMEOUT = 0x05  # Command not accepted in time


# =============================================================================
# Matmul Component
# =============================================================================


class Matmul(Component):
    """
    Matmul ISA instruction: C = A @ B + D

    The hardware automatically:
    - Selects optimal dataflow (OS/AS/BS) based on M, N, K dimensions
    - Tiles the computation to fit array dimensions
    - Double-buffers to hide memory latency
    - Generates internal command sequences

    Ports:
        Configuration Interface:
            cfg_valid: Configuration command available
            cfg_ready: Ready to accept configuration
            cfg_cmd: Configuration command code (MatmulCmd)
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
            progress_i, progress_j, progress_k: Current tile indices
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
            "progress_i": Out(16),
            "progress_j": Out(16),
            "progress_k": Out(16),
            "selected_dataflow": Out(2),  # 0=OS, 1=AS, 2=BS
            # Dimension outputs (for debugging)
            "param_M": Out(32),
            "param_N": Out(32),
            "param_K": Out(32),
            # Computed addresses (for debugging/verification)
            "dbg_addr_A": Out(64),
            "dbg_addr_B": Out(64),
            "dbg_addr_C": Out(64),
            "dbg_addr_D": Out(64),
            # Current tile sizes (for edge tile handling)
            "dbg_tile_M": Out(16),
            "dbg_tile_N": Out(16),
            "dbg_tile_K": Out(16),
            # Bank selection (for double buffering debugging)
            "dbg_sp_bank_A": Out(2),
            "dbg_sp_bank_B": Out(2),
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

        # Matrix dimensions
        reg_M = Signal(32, name="reg_M")
        reg_N = Signal(32, name="reg_N")
        reg_K = Signal(32, name="reg_K")

        # Base addresses
        reg_A_addr = Signal(64, name="reg_A_addr")
        reg_B_addr = Signal(64, name="reg_B_addr")
        reg_C_addr = Signal(64, name="reg_C_addr")
        reg_D_addr = Signal(64, name="reg_D_addr")

        # Strides
        reg_A_stride = Signal(32, name="reg_A_stride")
        reg_B_stride = Signal(32, name="reg_B_stride")
        reg_C_stride = Signal(32, name="reg_C_stride")
        reg_D_stride = Signal(32, name="reg_D_stride")

        # Options
        reg_accumulate = Signal(1, name="reg_accumulate")
        reg_activation = Signal(3, name="reg_activation")

        # Configuration valid flags
        cfg_dims_valid = Signal(1)
        cfg_addrs_valid = Signal(1)

        # =====================================================================
        # Tile Counters
        # =====================================================================

        # Number of tiles in each dimension
        tiles_M = Signal(16)  # ceil(M / dim)
        tiles_N = Signal(16)  # ceil(N / dim)
        tiles_K = Signal(16)  # ceil(K / dim)

        # Current tile indices
        tile_i = Signal(16)  # Output row tile
        tile_j = Signal(16)  # Output column tile
        tile_k = Signal(16)  # Inner dimension tile

        # Command ID counter
        cmd_id_counter = Signal(8)

        # =====================================================================
        # Address Calculation
        # =====================================================================

        # Element sizes in bytes
        input_bytes = cfg.input_bits // 8
        acc_bytes = cfg.acc_bits // 8

        # Computed tile addresses (active during LOAD/STORE states)
        # These are calculated combinationally from tile indices
        addr_A_tile = Signal(64, name="addr_A_tile")
        addr_B_tile = Signal(64, name="addr_B_tile")
        addr_C_tile = Signal(64, name="addr_C_tile")
        addr_D_tile = Signal(64, name="addr_D_tile")

        # Tile dimensions for current tile (may be smaller at edges)
        # These handle partial tiles when matrix dimensions aren't multiples of dim
        tile_M_size = Signal(16, name="tile_M_size")  # Rows in current A/C tile
        tile_N_size = Signal(16, name="tile_N_size")  # Cols in current B/C tile
        tile_K_size = Signal(16, name="tile_K_size")  # Cols in A / Rows in B

        # Row/column offsets for current tile (in elements)
        row_offset_i = Signal(32, name="row_offset_i")  # tile_i * dim
        col_offset_j = Signal(32, name="col_offset_j")  # tile_j * dim
        inner_offset_k = Signal(32, name="inner_offset_k")  # tile_k * dim

        # Compute element offsets from tile indices
        # Using shift for multiplication by dim (power of 2)
        dim_shift = (dim - 1).bit_length()  # log2(dim) for power-of-2 dim
        m.d.comb += [
            row_offset_i.eq(tile_i << dim_shift),
            col_offset_j.eq(tile_j << dim_shift),
            inner_offset_k.eq(tile_k << dim_shift),
        ]

        # Compute actual tile sizes (handle edge cases)
        # tile_size = min(dim, remaining_elements)
        remaining_M = Signal(32)
        remaining_N = Signal(32)
        remaining_K = Signal(32)

        m.d.comb += [
            remaining_M.eq(reg_M - row_offset_i),
            remaining_N.eq(reg_N - col_offset_j),
            remaining_K.eq(reg_K - inner_offset_k),
        ]

        # Clamp to dim (using Mux for min operation)
        m.d.comb += [
            tile_M_size.eq(Mux(remaining_M < dim, remaining_M[:16], dim)),
            tile_N_size.eq(Mux(remaining_N < dim, remaining_N[:16], dim)),
            tile_K_size.eq(Mux(remaining_K < dim, remaining_K[:16], dim)),
        ]

        # Address calculations:
        # A is M x K, row-major: A[row, col] at base + row * stride + col * input_bytes
        # A tile at (tile_i, tile_k): base + (tile_i * dim) * stride + (tile_k * dim) * input_bytes
        #
        # B is K x N, row-major: B[row, col] at base + row * stride + col * input_bytes
        # B tile at (tile_k, tile_j): base + (tile_k * dim) * stride + (tile_j * dim) * input_bytes
        #
        # C is M x N, row-major: C[row, col] at base + row * stride + col * acc_bytes
        # C tile at (tile_i, tile_j): base + (tile_i * dim) * stride + (tile_j * dim) * acc_bytes
        #
        # D has same layout as C

        # Intermediate products for address calculation
        # row_byte_offset = row_offset * stride
        # col_byte_offset = col_offset * element_bytes
        a_row_byte_offset = Signal(64, name="a_row_byte_offset")
        a_col_byte_offset = Signal(64, name="a_col_byte_offset")
        b_row_byte_offset = Signal(64, name="b_row_byte_offset")
        b_col_byte_offset = Signal(64, name="b_col_byte_offset")
        c_row_byte_offset = Signal(64, name="c_row_byte_offset")
        c_col_byte_offset = Signal(64, name="c_col_byte_offset")

        m.d.comb += [
            # A tile address components
            a_row_byte_offset.eq(row_offset_i * reg_A_stride),
            a_col_byte_offset.eq(inner_offset_k * input_bytes),
            addr_A_tile.eq(reg_A_addr + a_row_byte_offset + a_col_byte_offset),
            # B tile address components
            b_row_byte_offset.eq(inner_offset_k * reg_B_stride),
            b_col_byte_offset.eq(col_offset_j * input_bytes),
            addr_B_tile.eq(reg_B_addr + b_row_byte_offset + b_col_byte_offset),
            # C tile address components
            c_row_byte_offset.eq(row_offset_i * reg_C_stride),
            c_col_byte_offset.eq(col_offset_j * acc_bytes),
            addr_C_tile.eq(reg_C_addr + c_row_byte_offset + c_col_byte_offset),
            # D tile uses same calculation as C but with D base and stride
            addr_D_tile.eq(reg_D_addr + (row_offset_i * reg_D_stride) + (col_offset_j * acc_bytes)),
        ]

        # Scratchpad addresses for tiles (local memory addresses)
        # We use a simple allocation: A goes to bank 0, B goes to bank 1
        # For double buffering, we'll alternate between bank pairs
        sp_bank_A = Signal(2, name="sp_bank_A")  # 0 or 2
        sp_bank_B = Signal(2, name="sp_bank_B")  # 1 or 3
        acc_bank = Signal(1, name="acc_bank")  # 0 or 1

        # Local addresses within scratchpad (row-based)
        sp_addr_A = Signal(32, name="sp_addr_A")
        sp_addr_B = Signal(32, name="sp_addr_B")
        acc_addr_C = Signal(32, name="acc_addr_C")

        # Simple allocation: each tile gets dim rows starting at row 0
        # In a full implementation, this would be more sophisticated
        m.d.comb += [
            sp_addr_A.eq(sp_bank_A << 16),  # Bank in upper bits
            sp_addr_B.eq(sp_bank_B << 16),
            acc_addr_C.eq(acc_bank << 16),
        ]

        # =====================================================================
        # Transfer Length Calculation
        # =====================================================================

        # Number of rows to transfer for each tile (equals tile size)
        # Each row is one DMA beat for simplicity
        load_A_len = Signal(16, name="load_A_len")
        load_B_len = Signal(16, name="load_B_len")
        load_D_len = Signal(16, name="load_D_len")
        store_C_len = Signal(16, name="store_C_len")

        m.d.comb += [
            # A tile: tile_M_size rows, each row is dim elements
            load_A_len.eq(tile_M_size),
            # B tile: tile_K_size rows
            load_B_len.eq(tile_K_size),
            # D tile: tile_M_size rows (same shape as C)
            load_D_len.eq(tile_M_size),
            # C tile: tile_M_size rows
            store_C_len.eq(tile_M_size),
        ]

        # =====================================================================
        # Dataflow Selection
        # =====================================================================

        # Selected dataflow mode (computed from dimensions)
        dataflow_mode = Signal(2)  # 0=OS, 1=AS, 2=BS

        # Heuristic: pick based on which dimension is largest relative to others
        # - If K >> M, N: B-stationary (keep weights, stream activations)
        # - If M >> K, N: A-stationary (keep inputs, stream weights)
        # - Otherwise: Output-stationary (accumulate in place)
        with m.If(reg_K > (reg_M << 1)):
            with m.If(reg_K > (reg_N << 1)):
                # K is dominant -> B-stationary (weight stationary)
                m.d.comb += dataflow_mode.eq(2)
            with m.Else():
                m.d.comb += dataflow_mode.eq(0)  # Output-stationary
        with m.Elif(reg_M > (reg_K << 1)):
            with m.If(reg_M > (reg_N << 1)):
                # M is dominant -> A-stationary
                m.d.comb += dataflow_mode.eq(1)
            with m.Else():
                m.d.comb += dataflow_mode.eq(0)  # Output-stationary
        with m.Else():
            m.d.comb += dataflow_mode.eq(0)  # Output-stationary (default)

        # =====================================================================
        # State Machine
        # =====================================================================

        # State encoding
        # Note: We handle backpressure by staying in command-emitting states until
        # cmd_ready is asserted, so separate WAIT states are not needed.
        state = Signal(8)
        STATE_IDLE = 0
        STATE_INIT = 1
        STATE_CONFIG = 2
        STATE_LOAD_D = 3
        STATE_LOAD_A = 4
        STATE_LOAD_B = 5
        STATE_PRELOAD = 6
        STATE_COMPUTE = 7
        STATE_NEXT_K = 8
        STATE_STORE = 9
        STATE_NEXT_IJ = 10
        STATE_DONE = 11
        STATE_ERROR = 12

        # =====================================================================
        # Output Assignments
        # =====================================================================

        # Expose dimension parameters
        m.d.comb += [
            self.param_M.eq(reg_M),
            self.param_N.eq(reg_N),
            self.param_K.eq(reg_K),
            self.selected_dataflow.eq(dataflow_mode),
            self.progress_i.eq(tile_i),
            self.progress_j.eq(tile_j),
            self.progress_k.eq(tile_k),
        ]

        # Expose computed addresses for debugging
        m.d.comb += [
            self.dbg_addr_A.eq(addr_A_tile),
            self.dbg_addr_B.eq(addr_B_tile),
            self.dbg_addr_C.eq(addr_C_tile),
            self.dbg_addr_D.eq(addr_D_tile),
            self.dbg_tile_M.eq(tile_M_size),
            self.dbg_tile_N.eq(tile_N_size),
            self.dbg_tile_K.eq(tile_K_size),
            # Bank selection debug outputs
            self.dbg_sp_bank_A.eq(sp_bank_A),
            self.dbg_sp_bank_B.eq(sp_bank_B),
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
                    # Dimension configuration
                    with m.Case(MatmulCmd.CONFIG_DIMS):
                        m.d.sync += [
                            reg_M.eq(self.cfg_data[0:16]),
                            reg_N.eq(self.cfg_data[16:32]),
                            reg_K.eq(self.cfg_data[32:48]),
                            cfg_dims_valid.eq(1),
                        ]

                    # Address configuration
                    with m.Case(MatmulCmd.CONFIG_A_ADDR):
                        m.d.sync += reg_A_addr.eq(self.cfg_data)
                    with m.Case(MatmulCmd.CONFIG_B_ADDR):
                        m.d.sync += reg_B_addr.eq(self.cfg_data)
                    with m.Case(MatmulCmd.CONFIG_C_ADDR):
                        m.d.sync += [
                            reg_C_addr.eq(self.cfg_data),
                            cfg_addrs_valid.eq(1),
                        ]
                    with m.Case(MatmulCmd.CONFIG_D_ADDR):
                        m.d.sync += reg_D_addr.eq(self.cfg_data)

                    # Stride configuration
                    with m.Case(MatmulCmd.CONFIG_A_STRIDE):
                        m.d.sync += reg_A_stride.eq(self.cfg_data[0:32])
                    with m.Case(MatmulCmd.CONFIG_B_STRIDE):
                        m.d.sync += reg_B_stride.eq(self.cfg_data[0:32])
                    with m.Case(MatmulCmd.CONFIG_C_STRIDE):
                        m.d.sync += reg_C_stride.eq(self.cfg_data[0:32])
                    with m.Case(MatmulCmd.CONFIG_D_STRIDE):
                        m.d.sync += reg_D_stride.eq(self.cfg_data[0:32])

                    # Options
                    with m.Case(MatmulCmd.CONFIG_OPTIONS):
                        m.d.sync += [
                            reg_accumulate.eq(self.cfg_data[0]),
                            reg_activation.eq(self.cfg_data[1:4]),
                        ]

                    # Start execution
                    with m.Case(MatmulCmd.START):
                        # Validate configuration
                        with m.If(cfg_dims_valid & cfg_addrs_valid):
                            m.next = "INIT"
                        with m.Else():
                            m.d.sync += self.error_code.eq(MatmulError.CONFIG_INCOMPLETE)
                            m.next = "ERROR"

            # -----------------------------------------------------------------
            # INIT: Calculate tile counts and initialize counters
            # -----------------------------------------------------------------
            with m.State("INIT"):
                m.d.comb += state.eq(STATE_INIT)

                # Calculate number of tiles (ceiling division)
                # tiles = (size + dim - 1) / dim
                # For power-of-2 dim, we can use right shift by log2(dim)
                m.d.sync += [
                    tiles_M.eq((reg_M + dim - 1) >> dim_shift),
                    tiles_N.eq((reg_N + dim - 1) >> dim_shift),
                    tiles_K.eq((reg_K + dim - 1) >> dim_shift),
                    tile_i.eq(0),
                    tile_j.eq(0),
                    tile_k.eq(0),
                    cmd_id_counter.eq(0),
                    # Initialize bank selection for double buffering
                    sp_bank_A.eq(0),
                    sp_bank_B.eq(1),
                    acc_bank.eq(0),
                ]

                # Configure the execute controller first
                m.next = "CONFIG"

            # -----------------------------------------------------------------
            # CONFIG: Issue CONFIG command to set dataflow mode
            # -----------------------------------------------------------------
            with m.State("CONFIG"):
                m.d.comb += state.eq(STATE_CONFIG)

                # Emit EXEC_CONFIG command
                # rs1 = dataflow mode | (shift << 8)
                # The shift is typically 0 for integer matmul
                m.d.comb += [
                    self.cmd_valid.eq(1),
                    self.cmd_opcode.eq(InternalOpcode.EXEC_CONFIG),
                    self.cmd_rs1.eq(dataflow_mode),  # Dataflow: 0=OS, 1=AS, 2=BS
                ]

                # Wait for command to be accepted (backpressure handling)
                with m.If(self.cmd_ready):
                    # If bias is provided, load it first
                    with m.If(reg_D_addr != 0):
                        m.next = "LOAD_D"
                    with m.Else():
                        m.next = "LOAD_A"

            # -----------------------------------------------------------------
            # LOAD_D: Issue LOAD command for bias tile (D)
            # -----------------------------------------------------------------
            with m.State("LOAD_D"):
                m.d.comb += state.eq(STATE_LOAD_D)

                # Emit LOAD_D command
                # rs1 = DRAM source address (addr_D_tile)
                # rs2 = local destination address (accumulator)
                # rd = transfer length
                m.d.comb += [
                    self.cmd_valid.eq(1),
                    self.cmd_opcode.eq(InternalOpcode.LOAD_D),
                    self.cmd_rs1.eq(addr_D_tile),
                    self.cmd_rs2.eq(acc_addr_C),  # D goes to same place as C
                    self.cmd_rd.eq(load_D_len),
                ]

                # Wait for command to be accepted
                with m.If(self.cmd_ready):
                    m.next = "LOAD_A"

            # -----------------------------------------------------------------
            # LOAD_A: Issue LOAD command for A tile
            # -----------------------------------------------------------------
            with m.State("LOAD_A"):
                m.d.comb += state.eq(STATE_LOAD_A)

                # Emit LOAD_A command
                # rs1 = DRAM source address
                # rs2 = scratchpad destination address
                # rd = transfer length
                m.d.comb += [
                    self.cmd_valid.eq(1),
                    self.cmd_opcode.eq(InternalOpcode.LOAD_A),
                    self.cmd_rs1.eq(addr_A_tile),
                    self.cmd_rs2.eq(sp_addr_A),
                    self.cmd_rd.eq(load_A_len),
                ]

                # Wait for command to be accepted
                with m.If(self.cmd_ready):
                    m.next = "LOAD_B"

            # -----------------------------------------------------------------
            # LOAD_B: Issue LOAD command for B tile
            # -----------------------------------------------------------------
            with m.State("LOAD_B"):
                m.d.comb += state.eq(STATE_LOAD_B)

                # Emit LOAD_B command
                m.d.comb += [
                    self.cmd_valid.eq(1),
                    self.cmd_opcode.eq(InternalOpcode.LOAD_B),
                    self.cmd_rs1.eq(addr_B_tile),
                    self.cmd_rs2.eq(sp_addr_B),
                    self.cmd_rd.eq(load_B_len),
                ]

                # Wait for command to be accepted
                with m.If(self.cmd_ready):
                    # Only preload on first K iteration
                    with m.If(tile_k == 0):
                        m.next = "PRELOAD"
                    with m.Else():
                        m.next = "COMPUTE"

            # -----------------------------------------------------------------
            # PRELOAD: Issue PRELOAD command (first K iteration only)
            # -----------------------------------------------------------------
            with m.State("PRELOAD"):
                m.d.comb += state.eq(STATE_PRELOAD)

                # Emit EXEC_PRELOAD command
                # rs1 = accumulator source address (bias or zeros)
                # rs2 = number of rows
                m.d.comb += [
                    self.cmd_valid.eq(1),
                    self.cmd_opcode.eq(InternalOpcode.EXEC_PRELOAD),
                    self.cmd_rs1.eq(acc_addr_C),
                    self.cmd_rs2.eq(tile_M_size),
                ]

                # Wait for command to be accepted
                with m.If(self.cmd_ready):
                    m.next = "COMPUTE"

            # -----------------------------------------------------------------
            # COMPUTE: Issue COMPUTE command
            # -----------------------------------------------------------------
            with m.State("COMPUTE"):
                m.d.comb += state.eq(STATE_COMPUTE)

                # Emit EXEC_COMPUTE command
                # rs1 = A scratchpad address
                # rs2 = B scratchpad address
                # rd = C accumulator address
                # cmd_k_dim = inner dimension (tile_K_size)
                m.d.comb += [
                    self.cmd_valid.eq(1),
                    self.cmd_opcode.eq(InternalOpcode.EXEC_COMPUTE),
                    self.cmd_rs1.eq(sp_addr_A),
                    self.cmd_rs2.eq(sp_addr_B),
                    self.cmd_rd.eq(acc_addr_C),
                    self.cmd_k_dim.eq(tile_K_size),
                ]

                # Wait for command to be accepted
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
                        # Toggle scratchpad banks for double buffering
                        # Bank A: 0 <-> 2, Bank B: 1 <-> 3
                        sp_bank_A.eq(sp_bank_A ^ 2),
                        sp_bank_B.eq(sp_bank_B ^ 2),
                    ]
                    m.next = "LOAD_A"  # Load next A and B tiles
                with m.Else():
                    m.d.sync += tile_k.eq(0)
                    m.next = "STORE"

            # -----------------------------------------------------------------
            # STORE: Issue STORE command for C tile
            # -----------------------------------------------------------------
            with m.State("STORE"):
                m.d.comb += state.eq(STATE_STORE)

                # Emit STORE_C command
                # rs1 = accumulator source address
                # rs2 = DRAM destination address
                # rd = transfer length | (activation << 16)
                m.d.comb += [
                    self.cmd_valid.eq(1),
                    self.cmd_opcode.eq(InternalOpcode.STORE_C),
                    self.cmd_rs1.eq(acc_addr_C),
                    self.cmd_rs2.eq(addr_C_tile),
                    self.cmd_rd.eq(store_C_len | (reg_activation << 16)),
                ]

                # Wait for command to be accepted
                with m.If(self.cmd_ready):
                    m.next = "NEXT_IJ"

            # -----------------------------------------------------------------
            # NEXT_IJ: Advance I, J counters or finish
            # -----------------------------------------------------------------
            with m.State("NEXT_IJ"):
                m.d.comb += state.eq(STATE_NEXT_IJ)

                # Determine if we have more tiles to process
                has_more_j = Signal()
                has_more_i = Signal()
                has_bias = Signal()
                m.d.comb += [
                    has_more_j.eq(tile_j < tiles_N - 1),
                    has_more_i.eq(tile_i < tiles_M - 1),
                    has_bias.eq(reg_D_addr != 0),
                ]

                with m.If(has_more_j):
                    # Move to next column tile
                    m.d.sync += [
                        tile_j.eq(tile_j + 1),
                        # Toggle accumulator bank for double buffering
                        acc_bank.eq(~acc_bank),
                        # Reset scratchpad banks for new output tile
                        sp_bank_A.eq(0),
                        sp_bank_B.eq(1),
                    ]
                    with m.If(has_bias):
                        m.next = "LOAD_D"
                    with m.Else():
                        m.next = "LOAD_A"
                with m.Elif(has_more_i):
                    # Move to next row, reset column
                    m.d.sync += [
                        tile_i.eq(tile_i + 1),
                        tile_j.eq(0),
                        # Toggle accumulator bank for double buffering
                        acc_bank.eq(~acc_bank),
                        # Reset scratchpad banks for new output tile
                        sp_bank_A.eq(0),
                        sp_bank_B.eq(1),
                    ]
                    with m.If(has_bias):
                        m.next = "LOAD_D"
                    with m.Else():
                        m.next = "LOAD_A"
                with m.Else():
                    # All tiles processed
                    m.next = "DONE"

            # -----------------------------------------------------------------
            # DONE: Signal completion
            # -----------------------------------------------------------------
            with m.State("DONE"):
                m.d.comb += state.eq(STATE_DONE)
                # Stay in DONE until reset or new configuration
                # Return to IDLE after one cycle
                m.d.sync += [
                    cfg_dims_valid.eq(0),
                    cfg_addrs_valid.eq(0),
                ]
                m.next = "IDLE"

            # -----------------------------------------------------------------
            # ERROR: Signal error
            # -----------------------------------------------------------------
            with m.State("ERROR"):
                m.d.comb += state.eq(STATE_ERROR)
                # Return to IDLE after signaling error
                m.next = "IDLE"

        return m
