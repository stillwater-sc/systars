"""
StoreController - Orchestrates Accumulator -> DRAM transfers.

The StoreController handles MEMCPY commands for storing data by:
1. Accepting commands specifying accumulator source and DRAM destination
2. Reading data from accumulator
3. Issuing DMA write requests via StreamWriter
4. Signaling completion

State Machine:
    IDLE -> READ_ACC -> ISSUE_DMA -> SEND_DATA -> WAIT_DONE -> IDLE

Data Flow:
    Accumulator --> StoreController --> StreamWriter --> DRAM
"""

from amaranth import Module, Signal, signed
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig
from ..util.commands import DmaOpcode


class StoreController(Component):
    """
    Store controller for accumulator -> DRAM transfers.

    Coordinates data movement from the accumulator through the StreamWriter
    DMA engine to external memory.

    Ports:
        Command Interface:
            cmd_valid: Command available
            cmd_ready: Controller ready to accept command
            cmd_opcode: Operation code (MEMCPY)
            cmd_acc_addr: Source address in accumulator (32-bit local format)
            cmd_dram_addr: Destination address in DRAM (64-bit)
            cmd_len: Transfer length in beats
            cmd_id: Command ID for tracking
            cmd_activation: Activation function (0=none, 1=relu)

        Accumulator Interface:
            acc_read_req: Read request
            acc_read_addr: Read address (local format)
            acc_read_data: Read data (acc_width bits)
            acc_read_valid: Data valid

        DMA Interface (to StreamWriter):
            dma_req_valid: Request valid
            dma_req_ready: DMA ready to accept
            dma_req_addr: DRAM address (64-bit)
            dma_req_len: Burst length (AXI4 style: 0=1 beat)

            dma_data_valid: Write data valid
            dma_data_ready: DMA ready for data
            dma_data: Write data (dma_buswidth bits)
            dma_data_last: Last beat of burst

            dma_done: DMA write complete

        Status:
            busy: Controller is processing a command
            completed: Command completed this cycle
            completed_id: ID of completed command
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        buswidth = config.dma_buswidth
        acc_width = config.acc_width

        super().__init__(
            {
                # Command interface
                "cmd_valid": In(1),
                "cmd_ready": Out(1),
                "cmd_opcode": In(8),
                "cmd_acc_addr": In(32),
                "cmd_dram_addr": In(64),
                "cmd_len": In(16),  # Number of beats to transfer
                "cmd_id": In(8),
                "cmd_activation": In(4),  # Activation function
                # Accumulator read interface
                "acc_read_req": Out(1),
                "acc_read_addr": Out(32),
                "acc_read_data": In(signed(acc_width)),
                "acc_read_valid": In(1),
                # DMA interface (to StreamWriter)
                "dma_req_valid": Out(1),
                "dma_req_ready": In(1),
                "dma_req_addr": Out(64),
                "dma_req_len": Out(8),  # AXI4 style: 0=1 beat
                "dma_data_valid": Out(1),
                "dma_data_ready": In(1),
                "dma_data": Out(buswidth),
                "dma_data_last": Out(1),
                "dma_done": In(1),
                # Status
                "busy": Out(1),
                "completed": Out(1),
                "completed_id": Out(8),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        # State machine states
        STATE_IDLE = 0
        STATE_ISSUE_DMA = 1
        STATE_READ_ACC = 2
        STATE_WAIT_VALID = 3
        STATE_SEND_DATA = 4
        STATE_WAIT_DONE = 5
        STATE_DONE = 6

        state = Signal(3, init=STATE_IDLE)

        # Registered command parameters
        acc_addr_reg = Signal(32)
        dram_addr_reg = Signal(64)
        len_reg = Signal(16)
        cmd_id_reg = Signal(8)
        activation_reg = Signal(4)

        # Beat counter for accumulator address increment
        beat_count = Signal(16)

        # Data buffer for accumulator read
        data_buffer = Signal(cfg.acc_width)
        data_valid_buffer = Signal()

        # Default outputs
        m.d.comb += [
            self.cmd_ready.eq(0),
            self.acc_read_req.eq(0),
            self.acc_read_addr.eq(0),
            self.dma_req_valid.eq(0),
            self.dma_req_addr.eq(0),
            self.dma_req_len.eq(0),
            self.dma_data_valid.eq(0),
            self.dma_data.eq(0),
            self.dma_data_last.eq(0),
            self.busy.eq(state != STATE_IDLE),
            self.completed.eq(0),
            self.completed_id.eq(0),
        ]

        with m.Switch(state):
            with m.Case(STATE_IDLE):
                m.d.comb += self.cmd_ready.eq(1)

                # Accept MEMCPY commands for store operations
                with m.If(self.cmd_valid & (self.cmd_opcode == DmaOpcode.MEMCPY)):
                    m.d.sync += [
                        acc_addr_reg.eq(self.cmd_acc_addr),
                        dram_addr_reg.eq(self.cmd_dram_addr),
                        len_reg.eq(self.cmd_len),
                        cmd_id_reg.eq(self.cmd_id),
                        activation_reg.eq(self.cmd_activation),
                        beat_count.eq(0),
                        data_valid_buffer.eq(0),
                    ]
                    m.d.sync += state.eq(STATE_ISSUE_DMA)

            with m.Case(STATE_ISSUE_DMA):
                # Issue DMA write request to StreamWriter
                m.d.comb += [
                    self.dma_req_valid.eq(1),
                    self.dma_req_addr.eq(dram_addr_reg),
                    # AXI4 style: len=N means N+1 beats
                    self.dma_req_len.eq(len_reg - 1),
                ]

                with m.If(self.dma_req_ready):
                    m.d.sync += state.eq(STATE_READ_ACC)

            with m.Case(STATE_READ_ACC):
                # Request data from accumulator
                m.d.comb += [
                    self.acc_read_req.eq(1),
                    self.acc_read_addr.eq(acc_addr_reg + beat_count),
                ]
                m.d.sync += state.eq(STATE_WAIT_VALID)

            with m.Case(STATE_WAIT_VALID):  # noqa: SIM117
                # Wait for accumulator read valid
                with m.If(self.acc_read_valid):  # noqa: SIM117
                    # Buffer the data (apply activation if needed)
                    with m.If(activation_reg == 1):
                        # ReLU: max(0, x)
                        with m.If(self.acc_read_data >= 0):
                            m.d.sync += data_buffer.eq(self.acc_read_data)
                        with m.Else():
                            m.d.sync += data_buffer.eq(0)
                    with m.Else():
                        m.d.sync += data_buffer.eq(self.acc_read_data)

                    m.d.sync += data_valid_buffer.eq(1)
                    m.d.sync += state.eq(STATE_SEND_DATA)

            with m.Case(STATE_SEND_DATA):
                # Send data to DMA
                is_last = beat_count >= len_reg - 1
                m.d.comb += [
                    self.dma_data_valid.eq(1),
                    self.dma_data.eq(data_buffer),
                    self.dma_data_last.eq(is_last),
                ]

                with m.If(self.dma_data_ready):
                    m.d.sync += beat_count.eq(beat_count + 1)
                    m.d.sync += data_valid_buffer.eq(0)

                    with m.If(is_last):
                        m.d.sync += state.eq(STATE_WAIT_DONE)
                    with m.Else():
                        m.d.sync += state.eq(STATE_READ_ACC)

            with m.Case(STATE_WAIT_DONE):  # noqa: SIM117
                # Wait for DMA write to complete
                with m.If(self.dma_done):  # noqa: SIM117
                    m.d.sync += state.eq(STATE_DONE)

            with m.Case(STATE_DONE):
                # Signal completion
                m.d.comb += [
                    self.completed.eq(1),
                    self.completed_id.eq(cmd_id_reg),
                ]
                m.d.sync += state.eq(STATE_IDLE)

        return m
