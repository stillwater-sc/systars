"""
Matmul ISA instruction: C = A @ B

A hardware FSM that tiles a matrix multiply larger than the systolic array and
issues the LOAD / CONFIG / PRELOAD / COMPUTE / STORE commands for it, on the
command port ``SystolicTop`` takes (see ``isa.command``). ``connect_to_top``
wires it straight to a tile.

Dataflow: weight-stationary. Row-major tiles load one matrix row per scratchpad
row, which is the layout WS consumes: B tiles become the PE weights (PRELOAD)
and A tiles stream through them (COMPUTE).

For each output tile (i, j), and each K tile k:

- LOAD the A tile (rows of ``A[i*dim:, k*dim:]``) and the B tile
  (rows of ``B[k*dim:, j*dim:]``), one command per row at its exact address;
- PRELOAD the B tile as weights, and COMPUTE (the first K tile overwrites the
  accumulator rows, later ones accumulate);

then STORE the C tile, one command per row.

Layout requirements (checked at START; ``error_code`` UNSUPPORTED_LAYOUT
otherwise): the DMA moves whole bus beats, so every row that is loaded or stored
must start on a beat boundary. That means beat-aligned base addresses and
strides, and tiles past the first in K or N only when a tile row is a whole
number of beats. And whatever the DMA, the C stride must hold every N tile's
full row, because a STORE writes a whole tile row (dim results). Bias (``CONFIG_D_ADDR``)
and accumulating onto an existing C (``CONFIG_OPTIONS`` bit 0) both need a DMA into the
accumulator, which the tile does not have: UNSUPPORTED_BIAS and UNSUPPORTED_ACCUMULATE.

The tile's DMA beat is at least as wide as an accumulator row, so an int8 tile
row is never a whole number of beats: today N and K must each fit in one tile,
while M may span any number of tiles. ``require_beat_aligned=False`` skips the
layout check, for a DMA that can address any byte (and to verify the tiling
address arithmetic in N and K against a model).

Usage: configure dimensions, addresses and strides, then issue START.
"""

from enum import IntEnum

from amaranth import Module, Mux, Signal
from amaranth.lib.wiring import Component, In, Out

from ..config import Dataflow, SystolicConfig
from ..memory.local_addr import LocalAddr
from ..util.commands import DmaOpcode, ExecOpcode
from .command import CTRL_EXEC, CTRL_LOAD, CTRL_STORE, CommandIssuer, command_ports


class MatmulCmd(IntEnum):
    """Configuration commands for the Matmul instruction."""

    CONFIG_DIMS = 0x01  # data[15:0]=M, data[31:16]=N, data[47:32]=K
    CONFIG_A_ADDR = 0x10
    CONFIG_B_ADDR = 0x11
    CONFIG_C_ADDR = 0x12
    CONFIG_D_ADDR = 0x13  # Bias address, 0 = no bias (bias is not supported)
    CONFIG_A_STRIDE = 0x20  # Row strides in bytes
    CONFIG_B_STRIDE = 0x21
    CONFIG_C_STRIDE = 0x22
    CONFIG_D_STRIDE = 0x23
    CONFIG_OPTIONS = 0x30  # data[0]=accumulate, data[3:1]=activation
    START = 0xF0
    ABORT = 0xFF


class MatmulError(IntEnum):
    """Error codes for the Matmul instruction."""

    NONE = 0x00
    INVALID_DIMS = 0x01  # M, N or K is zero
    CONFIG_INCOMPLETE = 0x04  # START without dimensions and addresses
    UNSUPPORTED_LAYOUT = 0x06  # A row would not start on a DMA beat boundary
    UNSUPPORTED_BIAS = 0x07  # A bias (D) was configured
    UNSUPPORTED_ACCUMULATE = 0x08  # Accumulate onto C (C is not loaded into the accumulator)


WS = 1  # ExecuteController CONFIG rs1[0]


class Matmul(Component):
    """
    Matmul ISA instruction: C = A @ B.

    Ports:
        cfg_valid, cfg_ready, cfg_cmd, cfg_data: Configuration (``MatmulCmd``)
        cmd_*: Command port (``isa.command``)
        busy, done, error, error_code: Status
        progress_i, progress_j, progress_k: Current tile
        selected_dataflow: 1 (weight-stationary)
        param_M, param_N, param_K: Configured dimensions
        dbg_addr_A/B/C/D, dbg_tile_M/N/K: Current row addresses and tile sizes
    """

    def __init__(self, config: SystolicConfig, *, require_beat_aligned: bool = True):
        self.config = config
        self.require_beat_aligned = require_beat_aligned
        rows = config.grid_rows * config.tile_rows
        cols = config.grid_cols * config.tile_cols
        if rows != cols:
            raise ValueError(f"Matmul requires a square array, got {rows}x{cols}")
        self.dim = rows
        if self.dim & (self.dim - 1):
            # Tile offsets and counts are computed with shifts by log2(dim)
            raise ValueError(f"Matmul requires a power-of-two array dimension, got {self.dim}")
        if not config.dataflow & Dataflow.B_STATIONARY:
            raise ValueError("Matmul issues weight-stationary commands: the array must support WS")
        if config.sp_total_rows < 2 * self.dim or config.acc_total_rows < self.dim:
            raise ValueError("Matmul needs 2 * dim scratchpad rows and dim accumulator rows")

        addr = LocalAddr(config)
        # Scratchpad: A tile rows [0, dim), B tile rows [dim, 2*dim); accumulator: C rows [0, dim)
        self.sp_a = addr.make_sp_addr(0, 0)
        self.sp_b = addr.make_sp_addr(self.dim % config.sp_banks, self.dim // config.sp_banks)
        self.acc_c = addr.make_acc_addr(0, 0)

        ports = {
            "cfg_valid": In(1),
            "cfg_ready": Out(1),
            "cfg_cmd": In(8),
            "cfg_data": In(64),
            **command_ports(),
            "busy": Out(1),
            "done": Out(1),
            "error": Out(1),
            "error_code": Out(8),
            "progress_i": Out(16),
            "progress_j": Out(16),
            "progress_k": Out(16),
            "selected_dataflow": Out(2),
            "param_M": Out(32),
            "param_N": Out(32),
            "param_K": Out(32),
            "dbg_addr_A": Out(64),
            "dbg_addr_B": Out(64),
            "dbg_addr_C": Out(64),
            "dbg_addr_D": Out(64),
            "dbg_tile_M": Out(16),
            "dbg_tile_N": Out(16),
            "dbg_tile_K": Out(16),
        }
        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config
        dim = self.dim
        dim_shift = (dim - 1).bit_length()
        in_bytes = cfg.input_bits // 8
        acc_bytes = cfg.acc_bits // 8
        beat = cfg.dma_buswidth // 8
        issuer = CommandIssuer(m, self)

        reg_M, reg_N, reg_K = Signal(32), Signal(32), Signal(32)
        reg_A, reg_B, reg_C, reg_D = (Signal(64, name=f"reg_{n}_addr") for n in "ABCD")
        a_stride, b_stride, c_stride = (Signal(32, name=f"reg_{n}_stride") for n in "ABC")
        reg_accumulate = Signal()
        reg_activation = Signal(3)
        dims_valid = Signal()
        addrs_valid = Signal()

        tiles_M, tiles_N, tiles_K = Signal(16), Signal(16), Signal(16)
        tile_i, tile_j, tile_k = Signal(16), Signal(16), Signal(16)
        row = Signal(16)  # row within the tile being loaded or stored

        # Tile sizes (edge tiles are smaller)
        tile_M, tile_N, tile_K = Signal(16), Signal(16), Signal(16)
        for size, total, index in (
            (tile_M, reg_M, tile_i),
            (tile_N, reg_N, tile_j),
            (tile_K, reg_K, tile_k),
        ):
            remaining = Signal(32)
            m.d.comb += [
                remaining.eq(total - (index << dim_shift)),
                size.eq(Mux(remaining < dim, remaining[:16], dim)),
            ]

        # Row addresses
        a_row = Signal(64)
        b_row = Signal(64)
        c_row = Signal(64)
        m.d.comb += [
            a_row.eq(
                reg_A + ((tile_i << dim_shift) + row) * a_stride + (tile_k << dim_shift) * in_bytes
            ),
            b_row.eq(
                reg_B + ((tile_k << dim_shift) + row) * b_stride + (tile_j << dim_shift) * in_bytes
            ),
            c_row.eq(
                reg_C + ((tile_i << dim_shift) + row) * c_stride + (tile_j << dim_shift) * acc_bytes
            ),
        ]

        # Layout checks (see the module docstring)
        def aligned(value):
            return (value & (beat - 1)) == 0

        layout_ok = Signal()
        m.d.comb += layout_ok.eq(
            aligned(reg_A)
            & aligned(reg_B)
            & aligned(reg_C)
            & aligned(a_stride)
            & aligned(b_stride)
            & aligned(c_stride)
            & (a_stride != 0)
            & (b_stride != 0)
            & (c_stride != 0)
            & ((tiles_K <= 1) | ((dim * in_bytes) % beat == 0))
            & ((tiles_N <= 1) | (((dim * in_bytes) % beat == 0) & ((dim * acc_bytes) % beat == 0)))
        )
        # A STORE writes a whole tile row (dim results), so C rows must hold every
        # N tile in full, whatever the DMA
        c_stride_ok = Signal()
        m.d.comb += c_stride_ok.eq(c_stride >= (tiles_N << dim_shift) * acc_bytes)

        m.d.comb += [
            self.param_M.eq(reg_M),
            self.param_N.eq(reg_N),
            self.param_K.eq(reg_K),
            self.selected_dataflow.eq(WS),
            self.progress_i.eq(tile_i),
            self.progress_j.eq(tile_j),
            self.progress_k.eq(tile_k),
            self.dbg_addr_A.eq(a_row),
            self.dbg_addr_B.eq(b_row),
            self.dbg_addr_C.eq(c_row),
            self.dbg_addr_D.eq(reg_D),
            self.dbg_tile_M.eq(tile_M),
            self.dbg_tile_N.eq(tile_N),
            self.dbg_tile_K.eq(tile_K),
        ]

        with m.FSM(init="IDLE") as fsm:
            with m.State("IDLE"):
                m.d.comb += self.cfg_ready.eq(1)
                with m.If(self.cfg_valid), m.Switch(self.cfg_cmd):
                    with m.Case(MatmulCmd.CONFIG_DIMS):
                        m.d.sync += [
                            reg_M.eq(self.cfg_data[0:16]),
                            reg_N.eq(self.cfg_data[16:32]),
                            reg_K.eq(self.cfg_data[32:48]),
                            dims_valid.eq(1),
                        ]
                    with m.Case(MatmulCmd.CONFIG_A_ADDR):
                        m.d.sync += reg_A.eq(self.cfg_data)
                    with m.Case(MatmulCmd.CONFIG_B_ADDR):
                        m.d.sync += reg_B.eq(self.cfg_data)
                    with m.Case(MatmulCmd.CONFIG_C_ADDR):
                        m.d.sync += [reg_C.eq(self.cfg_data), addrs_valid.eq(1)]
                    with m.Case(MatmulCmd.CONFIG_D_ADDR):
                        m.d.sync += reg_D.eq(self.cfg_data)
                    with m.Case(MatmulCmd.CONFIG_A_STRIDE):
                        m.d.sync += a_stride.eq(self.cfg_data[0:32])
                    with m.Case(MatmulCmd.CONFIG_B_STRIDE):
                        m.d.sync += b_stride.eq(self.cfg_data[0:32])
                    with m.Case(MatmulCmd.CONFIG_C_STRIDE):
                        m.d.sync += c_stride.eq(self.cfg_data[0:32])
                    with m.Case(MatmulCmd.CONFIG_OPTIONS):
                        m.d.sync += [
                            reg_accumulate.eq(self.cfg_data[0]),
                            reg_activation.eq(self.cfg_data[1:4]),
                        ]
                    with m.Case(MatmulCmd.START):
                        with m.If(dims_valid & addrs_valid):
                            m.d.sync += [
                                tiles_M.eq((reg_M + dim - 1) >> dim_shift),
                                tiles_N.eq((reg_N + dim - 1) >> dim_shift),
                                tiles_K.eq((reg_K + dim - 1) >> dim_shift),
                                tile_i.eq(0),
                                tile_j.eq(0),
                                tile_k.eq(0),
                                row.eq(0),
                                *issuer.reset(),
                            ]
                            m.next = "VALIDATE"
                        with m.Else():
                            m.d.sync += self.error_code.eq(MatmulError.CONFIG_INCOMPLETE)
                            m.next = "ERROR"

            with m.State("VALIDATE"):
                with m.If((reg_M == 0) | (reg_N == 0) | (reg_K == 0)):
                    m.d.sync += self.error_code.eq(MatmulError.INVALID_DIMS)
                    m.next = "ERROR"
                with m.Elif(reg_D != 0):
                    m.d.sync += self.error_code.eq(MatmulError.UNSUPPORTED_BIAS)
                    m.next = "ERROR"
                with m.Elif(reg_accumulate):
                    m.d.sync += self.error_code.eq(MatmulError.UNSUPPORTED_ACCUMULATE)
                    m.next = "ERROR"
                with m.Elif(~c_stride_ok | (~layout_ok if self.require_beat_aligned else 0)):
                    m.d.sync += self.error_code.eq(MatmulError.UNSUPPORTED_LAYOUT)
                    m.next = "ERROR"
                with m.Else():
                    m.d.sync += self.error_code.eq(MatmulError.NONE)
                    m.next = "CONFIG"

            issuer.state("CONFIG", "LOAD_A", CTRL_EXEC, ExecOpcode.CONFIG, rs1=WS)

            # One LOAD per A tile row, then per B tile row
            issuer.state(
                "LOAD_A", "A_NEXT", CTRL_LOAD, DmaOpcode.MEMCPY,
                dram_addr=a_row, local_addr=self.sp_a + row, len=1,
            )  # fmt: skip
            with m.State("A_NEXT"):
                with m.If(row + 1 < tile_M):
                    m.d.sync += row.eq(row + 1)
                    m.next = "LOAD_A"
                with m.Else():
                    m.d.sync += row.eq(0)
                    m.next = "LOAD_B"
            issuer.state(
                "LOAD_B", "B_NEXT", CTRL_LOAD, DmaOpcode.MEMCPY,
                dram_addr=b_row, local_addr=self.sp_b + row, len=1,
            )  # fmt: skip
            with m.State("B_NEXT"):
                with m.If(row + 1 < tile_K):
                    m.d.sync += row.eq(row + 1)
                    m.next = "LOAD_B"
                with m.Else():
                    m.d.sync += row.eq(0)
                    m.next = "PRELOAD"

            issuer.state(
                "PRELOAD", "COMPUTE", CTRL_EXEC, ExecOpcode.PRELOAD, rs1=self.sp_b, k_dim=tile_K
            )
            issuer.state(
                "COMPUTE", "NEXT_K", CTRL_EXEC,
                Mux(tile_k == 0, ExecOpcode.COMPUTE, ExecOpcode.COMPUTE_ACCUMULATE),
                rs1=self.sp_a, rd=self.acc_c, k_dim=tile_M,
            )  # fmt: skip

            with m.State("NEXT_K"):
                with m.If(tile_k + 1 < tiles_K):
                    m.d.sync += tile_k.eq(tile_k + 1)
                    m.next = "LOAD_A"
                with m.Else():
                    m.d.sync += tile_k.eq(0)
                    m.next = "STORE"

            # One STORE per C tile row
            issuer.state(
                "STORE", "C_NEXT", CTRL_STORE, DmaOpcode.MEMCPY,
                dram_addr=c_row, local_addr=self.acc_c + row, len=1, activation=reg_activation,
            )  # fmt: skip
            with m.State("C_NEXT"):
                with m.If(row + 1 < tile_M):
                    m.d.sync += row.eq(row + 1)
                    m.next = "STORE"
                with m.Else():
                    m.d.sync += row.eq(0)
                    m.next = "NEXT_IJ"

            with m.State("NEXT_IJ"):
                with m.If(tile_j + 1 < tiles_N):
                    m.d.sync += tile_j.eq(tile_j + 1)
                    m.next = "LOAD_A"
                with m.Elif(tile_i + 1 < tiles_M):
                    m.d.sync += [tile_i.eq(tile_i + 1), tile_j.eq(0)]
                    m.next = "LOAD_A"
                with m.Else():
                    m.next = "DONE"

            with m.State("DONE"):
                m.d.comb += self.done.eq(1)
                m.d.sync += [dims_valid.eq(0), addrs_valid.eq(0)]
                m.next = "IDLE"

            with m.State("ERROR"):
                m.d.comb += self.error.eq(1)
                m.next = "IDLE"

        m.d.comb += self.busy.eq(~fsm.ongoing("IDLE"))
        return m
