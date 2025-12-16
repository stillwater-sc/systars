"""
LoadController - Orchestrates DRAM -> Scratchpad transfers.

The LoadController handles MEMCPY commands for loading data by:
1. Accepting commands specifying DRAM source and scratchpad destination
2. Issuing DMA read requests via StreamReader
3. Receiving data from StreamReader and writing to scratchpad
4. Signaling completion

State Machine:
    IDLE -> ISSUE_DMA -> RECV_DATA -> WRITE_SP -> CHECK_DONE -> IDLE

Data Flow:
    DRAM (via StreamReader) --> LoadController --> Scratchpad
"""

from amaranth import Module, Signal
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig
from ..util.commands import DmaOpcode


class LoadController(Component):
    """
    Load controller for DRAM -> scratchpad transfers.

    Coordinates data movement from external memory through the StreamReader
    DMA engine into the scratchpad memory.

    Ports:
        Command Interface:
            cmd_valid: Command available
            cmd_ready: Controller ready to accept command
            cmd_opcode: Operation code (MEMCPY)
            cmd_dram_addr: Source address in DRAM (64-bit)
            cmd_sp_addr: Destination address in scratchpad (32-bit local format)
            cmd_len: Transfer length in beats
            cmd_id: Command ID for tracking

        DMA Interface (to StreamReader):
            dma_req_valid: Request valid
            dma_req_ready: DMA ready to accept
            dma_req_addr: DRAM address (64-bit)
            dma_req_len: Burst length (AXI4 style: 0=1 beat)

            dma_resp_valid: Response data valid
            dma_resp_ready: Controller ready to accept
            dma_resp_data: Read data (dma_buswidth bits)
            dma_resp_last: Last beat of burst

        Scratchpad Interface:
            sp_write_en: Write enable
            sp_write_addr: Write address (local format)
            sp_write_data: Write data (sp_width bits)
            sp_write_mask: Byte mask for partial writes

        Status:
            busy: Controller is processing a command
            completed: Command completed this cycle
            completed_id: ID of completed command
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        buswidth = config.dma_buswidth
        sp_width = config.sp_width
        mask_width = sp_width // 8

        super().__init__(
            {
                # Command interface
                "cmd_valid": In(1),
                "cmd_ready": Out(1),
                "cmd_opcode": In(8),
                "cmd_dram_addr": In(64),
                "cmd_sp_addr": In(32),
                "cmd_len": In(16),  # Number of beats to transfer
                "cmd_id": In(8),
                # DMA interface (to StreamReader)
                "dma_req_valid": Out(1),
                "dma_req_ready": In(1),
                "dma_req_addr": Out(64),
                "dma_req_len": Out(8),  # AXI4 style: 0=1 beat
                "dma_resp_valid": In(1),
                "dma_resp_ready": Out(1),
                "dma_resp_data": In(buswidth),
                "dma_resp_last": In(1),
                # Scratchpad write interface
                "sp_write_en": Out(1),
                "sp_write_addr": Out(32),
                "sp_write_data": Out(sp_width),
                "sp_write_mask": Out(mask_width),
                # Status
                "busy": Out(1),
                "completed": Out(1),
                "completed_id": Out(8),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        # All bytes valid for full-width writes
        mask_all_ones = (1 << (cfg.sp_width // 8)) - 1

        # State machine states
        STATE_IDLE = 0
        STATE_ISSUE_DMA = 1
        STATE_RECV_DATA = 2
        STATE_DONE = 3

        state = Signal(2, init=STATE_IDLE)

        # Registered command parameters
        dram_addr_reg = Signal(64)
        sp_addr_reg = Signal(32)
        len_reg = Signal(16)
        cmd_id_reg = Signal(8)

        # Beat counter for scratchpad address increment
        beat_count = Signal(16)

        # Default outputs
        m.d.comb += [
            self.cmd_ready.eq(0),
            self.dma_req_valid.eq(0),
            self.dma_req_addr.eq(0),
            self.dma_req_len.eq(0),
            self.dma_resp_ready.eq(0),
            self.sp_write_en.eq(0),
            self.sp_write_addr.eq(0),
            self.sp_write_data.eq(0),
            self.sp_write_mask.eq(0),
            self.busy.eq(state != STATE_IDLE),
            self.completed.eq(0),
            self.completed_id.eq(0),
        ]

        with m.Switch(state):
            with m.Case(STATE_IDLE):
                m.d.comb += self.cmd_ready.eq(1)

                # Accept MEMCPY commands for load operations
                with m.If(self.cmd_valid & (self.cmd_opcode == DmaOpcode.MEMCPY)):
                    m.d.sync += [
                        dram_addr_reg.eq(self.cmd_dram_addr),
                        sp_addr_reg.eq(self.cmd_sp_addr),
                        len_reg.eq(self.cmd_len),
                        cmd_id_reg.eq(self.cmd_id),
                        beat_count.eq(0),
                    ]
                    m.d.sync += state.eq(STATE_ISSUE_DMA)

            with m.Case(STATE_ISSUE_DMA):
                # Issue DMA request to StreamReader
                m.d.comb += [
                    self.dma_req_valid.eq(1),
                    self.dma_req_addr.eq(dram_addr_reg),
                    # AXI4 style: len=N means N+1 beats, so subtract 1
                    # But we store actual beat count, so convert here
                    self.dma_req_len.eq(len_reg - 1),
                ]

                with m.If(self.dma_req_ready):
                    m.d.sync += state.eq(STATE_RECV_DATA)

            with m.Case(STATE_RECV_DATA):
                # Receive data from DMA and write to scratchpad
                m.d.comb += self.dma_resp_ready.eq(1)

                with m.If(self.dma_resp_valid):
                    # Write to scratchpad
                    m.d.comb += [
                        self.sp_write_en.eq(1),
                        self.sp_write_addr.eq(sp_addr_reg + beat_count),
                        self.sp_write_data.eq(self.dma_resp_data),
                        self.sp_write_mask.eq(mask_all_ones),
                    ]

                    m.d.sync += beat_count.eq(beat_count + 1)

                    with m.If(self.dma_resp_last):
                        m.d.sync += state.eq(STATE_DONE)

            with m.Case(STATE_DONE):
                # Signal completion
                m.d.comb += [
                    self.completed.eq(1),
                    self.completed_id.eq(cmd_id_reg),
                ]
                m.d.sync += state.eq(STATE_IDLE)

        return m
