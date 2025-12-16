"""
ExecuteController - Orchestrates matrix multiply operations on the systolic array.

The ExecuteController manages the data flow for systolic matmul operations:
- CONFIG: Set dataflow mode and shift amount
- PRELOAD: Load bias/initial values from accumulator into array
- COMPUTE: Execute matmul by feeding A and B from scratchpad, storing C to accumulator

State Machine:
    IDLE -> CONFIG -> IDLE (on CONFIG command)
    IDLE -> PRELOAD_START -> PRELOAD_WAIT -> PRELOAD_FEED -> IDLE (on PRELOAD)
    IDLE -> COMPUTE_START -> COMPUTE_A_WAIT -> COMPUTE_B_WAIT -> COMPUTE_FEED
         -> COMPUTE_LOOP -> COMPUTE_FLUSH -> COMPUTE_WRITE -> IDLE (on COMPUTE)

Data Flow:
    Scratchpad (A, B) --> SystolicArray --> Accumulator (C)
                              ^
                              |
    Accumulator (D) --(preload bias)

For a simple 1x1 matmul (single row × single column), the flow is:
1. PRELOAD: Read D[col] from accumulator, feed to array in_d[col]
2. COMPUTE: Read A[row] from scratchpad, B[col] from scratchpad
            Feed to array: in_a[row], in_b[col]
            Wait for pipeline latency
            Read out_c[col] from array, write to accumulator
"""

from amaranth import Module, Signal, signed
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig
from ..util.commands import OpCode


class ExecuteController(Component):
    """
    Execute controller for systolic array operations.

    Coordinates data movement between scratchpad, accumulator, and systolic array
    to perform matrix multiply operations.

    Ports:
        Command Interface:
            cmd_valid: Command available
            cmd_ready: Controller ready to accept command
            cmd_opcode: Operation code (CONFIG, PRELOAD, COMPUTE)
            cmd_rs1: Source operand 1 (A address for COMPUTE, acc addr for PRELOAD)
            cmd_rs2: Source operand 2 (B address for COMPUTE)
            cmd_rd: Destination (C address for COMPUTE)
            cmd_config: Configuration data (dataflow, shift for CONFIG)

        Scratchpad Interface:
            sp_read_req: Request read from scratchpad
            sp_read_addr: Scratchpad address
            sp_read_data: Read data (sp_width bits)
            sp_read_valid: Data is valid

        Accumulator Interface:
            acc_read_req: Request read from accumulator
            acc_read_addr: Accumulator address
            acc_read_data: Read data (acc_width bits)
            acc_read_valid: Data is valid
            acc_write_req: Request write to accumulator
            acc_write_addr: Write address
            acc_write_data: Write data
            acc_accumulate: Add to existing value (vs overwrite)

        Status:
            busy: Controller is processing a command
            completed: Command completed this cycle
            completed_id: ID of completed command
    """

    def __init__(self, config: SystolicConfig):
        self.config = config

        # Calculate dimensions
        total_rows = config.mesh_rows * config.tile_rows
        total_cols = config.mesh_cols * config.tile_cols

        ports = {
            # Command interface (simplified)
            "cmd_valid": In(1),
            "cmd_ready": Out(1),
            "cmd_opcode": In(8),
            "cmd_rs1": In(32),  # A address or preload source
            "cmd_rs2": In(32),  # B address
            "cmd_rd": In(32),  # C destination address
            "cmd_k_dim": In(16),  # Inner dimension for matmul
            "cmd_id": In(8),  # Command ID for tracking
            # Configuration (set via CONFIG command or directly)
            "cfg_dataflow": Out(1),  # 0=OS, 1=WS
            "cfg_shift": Out(5),  # Rounding shift amount
            "cfg_propagate": Out(1),  # Which register bank to use
            # Scratchpad read interface (two ports for A and B)
            "sp_a_read_req": Out(1),
            "sp_a_read_addr": Out(32),
            "sp_a_read_data": In(signed(config.sp_width)),
            "sp_a_read_valid": In(1),
            "sp_b_read_req": Out(1),
            "sp_b_read_addr": Out(32),
            "sp_b_read_data": In(signed(config.sp_width)),
            "sp_b_read_valid": In(1),
            # Accumulator read interface (for preload)
            "acc_read_req": Out(1),
            "acc_read_addr": Out(32),
            "acc_read_data": In(signed(config.acc_width)),
            "acc_read_valid": In(1),
            # Accumulator write interface (for results)
            "acc_write_req": Out(1),
            "acc_write_addr": Out(32),
            "acc_write_data": Out(signed(config.acc_width)),
            "acc_accumulate": Out(1),
            # Status
            "busy": Out(1),
            "completed": Out(1),
            "completed_id": Out(8),
            "state_debug": Out(8),  # For debugging
        }

        # Systolic array interface - vector signals
        # Input vectors to array
        for i in range(total_rows):
            ports[f"array_in_a_{i}"] = Out(signed(config.input_bits))
        for j in range(total_cols):
            ports[f"array_in_b_{j}"] = Out(signed(config.weight_bits))
            ports[f"array_in_d_{j}"] = Out(signed(config.acc_bits))

        # Array control outputs
        ports["array_in_valid"] = Out(1)
        ports["array_in_control_dataflow"] = Out(1)
        ports["array_in_control_propagate"] = Out(1)
        ports["array_in_control_shift"] = Out(5)
        ports["array_in_id"] = Out(8)
        ports["array_in_last"] = Out(1)

        # Output vectors from array
        for j in range(total_cols):
            ports[f"array_out_c_{j}"] = In(signed(config.acc_bits))

        # Array status inputs
        ports["array_out_valid"] = In(1)
        ports["array_out_id"] = In(8)
        ports["array_out_last"] = In(1)

        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        total_rows = cfg.mesh_rows * cfg.tile_rows
        total_cols = cfg.mesh_cols * cfg.tile_cols

        # Pipeline latency through the array
        # For a mesh, data takes mesh_rows + mesh_cols - 1 cycles to traverse
        array_latency = cfg.mesh_rows + cfg.mesh_cols - 1

        # =================================================================
        # Internal Registers
        # =================================================================

        # Configuration registers
        dataflow_reg = Signal(1, init=0)
        shift_reg = Signal(5, init=0)
        propagate_reg = Signal(1, init=0)

        # Command registers (latched on command accept)
        cmd_id_reg = Signal(8)
        a_addr_reg = Signal(32)
        b_addr_reg = Signal(32)
        c_addr_reg = Signal(32)
        k_dim_reg = Signal(16)

        # Loop counters
        k_count = Signal(16)  # Current k iteration
        flush_count = Signal(16)  # Flush cycle counter

        # State encoding
        STATE_IDLE = 0
        STATE_CONFIG = 1
        STATE_PRELOAD_START = 2
        STATE_PRELOAD_WAIT = 3
        STATE_PRELOAD_FEED = 4
        STATE_COMPUTE_START = 5
        STATE_COMPUTE_A_REQ = 6
        STATE_COMPUTE_B_REQ = 7
        STATE_COMPUTE_WAIT = 8
        STATE_COMPUTE_FEED = 9
        STATE_COMPUTE_FLUSH = 10
        STATE_COMPUTE_WRITE = 11

        state = Signal(8, init=STATE_IDLE)

        # =================================================================
        # Output Configuration
        # =================================================================

        m.d.comb += [
            self.cfg_dataflow.eq(dataflow_reg),
            self.cfg_shift.eq(shift_reg),
            self.cfg_propagate.eq(propagate_reg),
            self.state_debug.eq(state),
        ]

        # =================================================================
        # Default Signal Values
        # =================================================================

        m.d.comb += [
            # Command interface
            self.cmd_ready.eq(state == STATE_IDLE),
            # Scratchpad interface
            self.sp_a_read_req.eq(0),
            self.sp_a_read_addr.eq(0),
            self.sp_b_read_req.eq(0),
            self.sp_b_read_addr.eq(0),
            # Accumulator interface
            self.acc_read_req.eq(0),
            self.acc_read_addr.eq(0),
            self.acc_write_req.eq(0),
            self.acc_write_addr.eq(0),
            self.acc_write_data.eq(0),
            self.acc_accumulate.eq(0),
            # Array interface - defaults
            self.array_in_valid.eq(0),
            self.array_in_control_dataflow.eq(dataflow_reg),
            self.array_in_control_propagate.eq(propagate_reg),
            self.array_in_control_shift.eq(shift_reg),
            self.array_in_id.eq(cmd_id_reg),
            self.array_in_last.eq(0),
            # Status
            self.busy.eq(state != STATE_IDLE),
            self.completed.eq(0),
            self.completed_id.eq(0),
        ]

        # Default array inputs to zero
        for i in range(total_rows):
            m.d.comb += getattr(self, f"array_in_a_{i}").eq(0)
        for j in range(total_cols):
            m.d.comb += getattr(self, f"array_in_b_{j}").eq(0)
            m.d.comb += getattr(self, f"array_in_d_{j}").eq(0)

        # =================================================================
        # State Machine
        # =================================================================

        with m.FSM(init="IDLE"):
            # ---------------------------------------------------------
            # IDLE: Wait for command
            # ---------------------------------------------------------
            with m.State("IDLE"):
                m.d.comb += state.eq(STATE_IDLE)

                with m.If(self.cmd_valid):
                    # Latch command parameters
                    m.d.sync += [
                        cmd_id_reg.eq(self.cmd_id),
                        a_addr_reg.eq(self.cmd_rs1),
                        b_addr_reg.eq(self.cmd_rs2),
                        c_addr_reg.eq(self.cmd_rd),
                        k_dim_reg.eq(self.cmd_k_dim),
                    ]

                    # Dispatch based on opcode
                    with m.Switch(self.cmd_opcode):
                        with m.Case(OpCode.CONFIG_EX):
                            m.next = "CONFIG"
                        with m.Case(OpCode.PRELOAD):
                            m.next = "PRELOAD_START"
                        with m.Case(OpCode.COMPUTE):
                            m.next = "COMPUTE_START"
                        with m.Default():
                            m.next = "IDLE"  # Unknown opcode, ignore

            # ---------------------------------------------------------
            # CONFIG: Apply configuration
            # ---------------------------------------------------------
            with m.State("CONFIG"):
                m.d.comb += state.eq(STATE_CONFIG)

                # Extract config from cmd_rs1:
                # [0]: dataflow, [4:1]: shift
                m.d.sync += [
                    dataflow_reg.eq(self.cmd_rs1[0]),
                    shift_reg.eq(self.cmd_rs1[1:6]),
                ]

                # Signal completion
                m.d.comb += [
                    self.completed.eq(1),
                    self.completed_id.eq(cmd_id_reg),
                ]

                m.next = "IDLE"

            # ---------------------------------------------------------
            # PRELOAD: Load bias from accumulator to array
            # ---------------------------------------------------------
            with m.State("PRELOAD_START"):
                m.d.comb += state.eq(STATE_PRELOAD_START)

                # Request read from accumulator
                m.d.comb += [
                    self.acc_read_req.eq(1),
                    self.acc_read_addr.eq(a_addr_reg),  # rs1 has acc address
                ]

                m.d.sync += k_count.eq(0)
                m.next = "PRELOAD_WAIT"

            with m.State("PRELOAD_WAIT"):
                m.d.comb += state.eq(STATE_PRELOAD_WAIT)

                # Wait for accumulator read valid
                with m.If(self.acc_read_valid):
                    m.next = "PRELOAD_FEED"

            with m.State("PRELOAD_FEED"):
                m.d.comb += state.eq(STATE_PRELOAD_FEED)

                # Feed bias values to array D inputs
                # For simplicity, broadcast same value to all columns
                # In a full implementation, would unpack acc_read_data
                m.d.comb += self.array_in_valid.eq(1)

                # Extract individual elements from acc_read_data
                # Each element is acc_bits wide
                for j in range(total_cols):
                    start_bit = j * cfg.acc_bits
                    end_bit = start_bit + cfg.acc_bits
                    m.d.comb += getattr(self, f"array_in_d_{j}").eq(
                        self.acc_read_data[start_bit:end_bit]
                    )

                # Signal completion
                m.d.comb += [
                    self.completed.eq(1),
                    self.completed_id.eq(cmd_id_reg),
                ]

                m.next = "IDLE"

            # ---------------------------------------------------------
            # COMPUTE: Execute matmul
            # ---------------------------------------------------------
            with m.State("COMPUTE_START"):
                m.d.comb += state.eq(STATE_COMPUTE_START)

                m.d.sync += [
                    k_count.eq(0),
                    flush_count.eq(0),
                ]

                m.next = "COMPUTE_A_REQ"

            with m.State("COMPUTE_A_REQ"):
                m.d.comb += state.eq(STATE_COMPUTE_A_REQ)

                # Request A data from scratchpad
                m.d.comb += [
                    self.sp_a_read_req.eq(1),
                    self.sp_a_read_addr.eq(a_addr_reg + k_count),
                ]

                m.next = "COMPUTE_B_REQ"

            with m.State("COMPUTE_B_REQ"):
                m.d.comb += state.eq(STATE_COMPUTE_B_REQ)

                # Keep A request active, also request B
                m.d.comb += [
                    self.sp_a_read_req.eq(1),
                    self.sp_a_read_addr.eq(a_addr_reg + k_count),
                    self.sp_b_read_req.eq(1),
                    self.sp_b_read_addr.eq(b_addr_reg + k_count),
                ]

                m.next = "COMPUTE_WAIT"

            with m.State("COMPUTE_WAIT"):
                m.d.comb += state.eq(STATE_COMPUTE_WAIT)

                # Wait for both A and B to be valid
                with m.If(self.sp_a_read_valid & self.sp_b_read_valid):
                    m.next = "COMPUTE_FEED"

            with m.State("COMPUTE_FEED"):
                m.d.comb += state.eq(STATE_COMPUTE_FEED)

                # Feed data to array
                m.d.comb += self.array_in_valid.eq(1)

                # Unpack A data (one element per row)
                for i in range(total_rows):
                    start_bit = i * cfg.input_bits
                    end_bit = start_bit + cfg.input_bits
                    m.d.comb += getattr(self, f"array_in_a_{i}").eq(
                        self.sp_a_read_data[start_bit:end_bit]
                    )

                # Unpack B data (one element per column)
                for j in range(total_cols):
                    start_bit = j * cfg.weight_bits
                    end_bit = start_bit + cfg.weight_bits
                    m.d.comb += getattr(self, f"array_in_b_{j}").eq(
                        self.sp_b_read_data[start_bit:end_bit]
                    )

                # Set last flag on final k iteration
                with m.If(k_count >= k_dim_reg - 1):
                    m.d.comb += self.array_in_last.eq(1)
                    m.next = "COMPUTE_FLUSH"
                with m.Else():
                    m.d.sync += k_count.eq(k_count + 1)
                    m.next = "COMPUTE_A_REQ"

            with m.State("COMPUTE_FLUSH"):
                m.d.comb += state.eq(STATE_COMPUTE_FLUSH)

                # Wait for array pipeline to drain
                m.d.sync += flush_count.eq(flush_count + 1)

                with m.If(flush_count >= array_latency):
                    m.next = "COMPUTE_WRITE"

            with m.State("COMPUTE_WRITE"):
                m.d.comb += state.eq(STATE_COMPUTE_WRITE)

                # Write results to accumulator
                with m.If(self.array_out_valid):
                    # Pack output data from array
                    acc_data = Signal(cfg.acc_width)
                    for j in range(total_cols):
                        start_bit = j * cfg.acc_bits
                        end_bit = start_bit + cfg.acc_bits
                        m.d.comb += acc_data[start_bit:end_bit].eq(
                            getattr(self, f"array_out_c_{j}")
                        )

                    m.d.comb += [
                        self.acc_write_req.eq(1),
                        self.acc_write_addr.eq(c_addr_reg),
                        self.acc_write_data.eq(acc_data),
                        self.acc_accumulate.eq(1),  # Accumulate mode
                    ]

                # Wait for last output
                with m.If(self.array_out_last):
                    m.d.comb += [
                        self.completed.eq(1),
                        self.completed_id.eq(cmd_id_reg),
                    ]
                    m.next = "IDLE"

        return m
