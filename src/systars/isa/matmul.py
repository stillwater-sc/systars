"""
Matmul ISA instruction: C = A @ B

A hardware FSM that tiles a matrix multiply larger than the systolic array and
issues the LOAD / CONFIG / PRELOAD / COMPUTE / STORE commands for it, on the
command port ``SystolicTop`` takes (see ``isa.command``). ``connect_to_top``
wires it straight to a tile.

Dataflow (``CONFIG_DATAFLOW``; default OS when the array has it, else WS, else
IS). The BlockMover's corner turn (``DmaOpcode.TRANSPOSE``) lays each tile out in
L2 as the Streamer streams it:

- OS: the A tile row-major (MEMCPY) and the B tile column-major (TRANSPOSE); one
  COMPUTE per K tile, the K tiles chained into one reduction held in the PEs
  (``cmd_chain``), so with K >= 2 * dim - 1 the tiles stream with no bubbles.
- WS: the A tile column-major (TRANSPOSE) streams through the B tile, row-major
  (MEMCPY), preloaded as the stationary tile; later K tiles accumulate.
- IS: the same loads, but A (as Aᵀ) is the stationary tile and B streams; the
  accumulator holds Cᵀ, and the STORE's corner turn writes C.

For each output tile (i, j), and each K tile k:

- LOAD the A tile (``A[i*dim:, k*dim:]``) and the B tile (``B[k*dim:, j*dim:]``),
  one command per tile: the BlockMover moves its rows from any byte address,
  ``stride`` bytes apart, with ``row_bytes`` valid bytes (an edge tile's rows are
  zero-filled past them);
- COMPUTE as the dataflow does (above);

then STORE the C tile, one command: its rows are written with byte strobes, so an
edge N tile writes only its own columns and C can be dense.

The commands are pushed without waiting for each other (``isa.command``): credits
and tile-ready tokens order them in ``SystolicTop``. Each K tile uses the next A/B
pair of L2 slots and each C tile the next accumulator slot, in ring order
(``TileBuffers``), so with ``buffers=2`` (the default) the BlockMover loads the next
tiles while the Streamer works on the current ones; ``buffers=1`` serializes them,
with the same results. The instruction is done when its last STORE completes.

So A, B and C may be dense row-major at any address, with any M, N and K. Errors
(``error_code``): UNSUPPORTED_DATAFLOW for a dataflow the array was not built
with; UNSUPPORTED_LAYOUT for a zero stride, or a C stride below N
results (rows would overlap). Bias (``CONFIG_D_ADDR``) and accumulating onto an
existing C (``CONFIG_OPTIONS`` bit 0) both need a load into the accumulator, which
the tile does not have yet: UNSUPPORTED_BIAS and UNSUPPORTED_ACCUMULATE.

Usage: configure dimensions, addresses and strides, then issue START.
"""

from enum import IntEnum

from amaranth import Module, Mux, Signal
from amaranth.lib.wiring import Component, In, Out

from ..config import Dataflow, SystolicConfig
from ..util.commands import DmaOpcode, ExecOpcode
from .command import CTRL_EXEC, CTRL_LOAD, CTRL_STORE, CommandIssuer, TileBuffers, command_ports


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
    CONFIG_DATAFLOW = 0x31  # data[1:0]: 0 = OS, 1 = WS, 2 = IS (default: OS if built, else WS)
    START = 0xF0
    ABORT = 0xFF


class MatmulError(IntEnum):
    """Error codes for the Matmul instruction."""

    NONE = 0x00
    INVALID_DIMS = 0x01  # M, N or K is zero
    CONFIG_INCOMPLETE = 0x04  # START without dimensions and addresses
    UNSUPPORTED_LAYOUT = 0x06  # A zero stride, or C rows that would overlap
    UNSUPPORTED_BIAS = 0x07  # A bias (D) was configured
    UNSUPPORTED_ACCUMULATE = 0x08  # Accumulate onto C (C is not loaded into the accumulator)
    UNSUPPORTED_DATAFLOW = 0x09  # The array was not built with the selected dataflow


OS, WS, IS = 0, 1, 2  # CONFIG_DATAFLOW, and the ExecuteController CONFIG rs1


class Matmul(Component):
    """
    Matmul ISA instruction: C = A @ B.

    Ports:
        cfg_valid, cfg_ready, cfg_cmd, cfg_data: Configuration (``MatmulCmd``)
        cmd_*: Command port (``isa.command``)
        busy, done, error, error_code: Status
        progress_i, progress_j, progress_k: Current tile
        selected_dataflow: The dataflow of the next START (0 = OS, 1 = WS, 2 = IS)
        param_M, param_N, param_K: Configured dimensions
        dbg_addr_A/B/C/D, dbg_tile_M/N/K: Current tile addresses and tile sizes
    """

    def __init__(self, config: SystolicConfig, buffers: int = 2):
        self.config = config
        rows = config.grid_rows * config.tile_rows
        cols = config.grid_cols * config.tile_cols
        if rows != cols:
            raise ValueError(f"Matmul requires a square array, got {rows}x{cols}")
        self.dim = rows
        if self.dim & (self.dim - 1):
            # Tile offsets and counts are computed with shifts by log2(dim)
            raise ValueError(f"Matmul requires a power-of-two array dimension, got {self.dim}")
        self.supports = {
            OS: bool(config.dataflow & Dataflow.OUTPUT_STATIONARY),
            WS: bool(config.dataflow & Dataflow.B_STATIONARY),
            IS: bool(config.dataflow & Dataflow.A_STATIONARY),
        }
        if not any(self.supports.values()):
            raise ValueError(f"Matmul needs an OS, WS or IS array, got {config.dataflow!r}")
        self.default_dataflow = OS if self.supports[OS] else (WS if self.supports[WS] else IS)
        self.tiles = TileBuffers(config, self.dim, buffers, "Matmul")

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
        issuer = CommandIssuer(m, self)
        bufs = self.tiles.build(m)

        reg_M, reg_N, reg_K = Signal(32), Signal(32), Signal(32)
        reg_A, reg_B, reg_C, reg_D = (Signal(64, name=f"reg_{n}_addr") for n in "ABCD")
        a_stride, b_stride, c_stride = (Signal(32, name=f"reg_{n}_stride") for n in "ABC")
        reg_accumulate = Signal()
        reg_activation = Signal(3)
        df = Signal(2, init=self.default_dataflow)  # dataflow of the next START
        df_ok = Signal()
        m.d.comb += df_ok.eq(
            ((df == OS) & self.supports[OS])
            | ((df == WS) & self.supports[WS])
            | ((df == IS) & self.supports[IS])
        )
        is_os = df == OS
        dims_valid = Signal()
        addrs_valid = Signal()

        tiles_M, tiles_N, tiles_K = Signal(16), Signal(16), Signal(16)
        tile_i, tile_j, tile_k = Signal(16), Signal(16), Signal(16)

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

        # Tile addresses (row 0 of the tile)
        a_tile = Signal(64)
        b_tile = Signal(64)
        c_tile = Signal(64)
        m.d.comb += [
            a_tile.eq(reg_A + (tile_i << dim_shift) * a_stride + (tile_k << dim_shift) * in_bytes),
            b_tile.eq(reg_B + (tile_k << dim_shift) * b_stride + (tile_j << dim_shift) * in_bytes),
            c_tile.eq(reg_C + (tile_i << dim_shift) * c_stride + (tile_j << dim_shift) * acc_bytes),
        ]

        # Layout check: rows a stride apart (0 would mean one bus beat to the
        # BlockMover), and C rows that do not overlap
        layout_ok = Signal()
        m.d.comb += layout_ok.eq(
            (a_stride != 0) & (b_stride != 0) & (c_stride >= reg_N * acc_bytes)
        )

        m.d.comb += [
            self.param_M.eq(reg_M),
            self.param_N.eq(reg_N),
            self.param_K.eq(reg_K),
            self.selected_dataflow.eq(df),
            self.progress_i.eq(tile_i),
            self.progress_j.eq(tile_j),
            self.progress_k.eq(tile_k),
            self.dbg_addr_A.eq(a_tile),
            self.dbg_addr_B.eq(b_tile),
            self.dbg_addr_C.eq(c_tile),
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
                    with m.Case(MatmulCmd.CONFIG_DATAFLOW):
                        m.d.sync += df.eq(self.cfg_data[0:2])
                    with m.Case(MatmulCmd.START):
                        with m.If(dims_valid & addrs_valid):
                            m.d.sync += [
                                tiles_M.eq((reg_M + dim - 1) >> dim_shift),
                                tiles_N.eq((reg_N + dim - 1) >> dim_shift),
                                tiles_K.eq((reg_K + dim - 1) >> dim_shift),
                                tile_i.eq(0),
                                tile_j.eq(0),
                                tile_k.eq(0),
                                *issuer.reset(),
                                *bufs.reset(),
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
                with m.Elif(~layout_ok):
                    m.d.sync += self.error_code.eq(MatmulError.UNSUPPORTED_LAYOUT)
                    m.next = "ERROR"
                with m.Elif(~df_ok):
                    m.d.sync += self.error_code.eq(MatmulError.UNSUPPORTED_DATAFLOW)
                    m.next = "ERROR"
                with m.Else():
                    m.d.sync += self.error_code.eq(MatmulError.NONE)
                    m.next = "CONFIG"

            issuer.state("CONFIG", "LOAD_A", CTRL_EXEC, ExecOpcode.CONFIG, rs1=df)

            first_k = tile_k == 0
            last_k = tile_k + 1 >= tiles_K

            # One LOAD per tile: the A tile (tile_M rows of tile_K), then the B tile
            # (tile_K rows of tile_N). The corner turns lay them out as the Streamer
            # streams them: OS takes A row-major and B column-major, WS and IS A
            # column-major and B row-major. OS tracks the pair with the A slot.
            issuer.state(
                "LOAD_A", "LOAD_B", CTRL_LOAD, Mux(is_os, DmaOpcode.MEMCPY, DmaOpcode.TRANSPOSE),
                dram_addr=a_tile, local_addr=bufs.a_addr, len=tile_M,
                stride=a_stride, row_bytes=tile_K * in_bytes,
                sync=Mux(is_os, bufs.load(bufs.a_slot, last=False), bufs.load(bufs.a_slot)),
            )  # fmt: skip
            with m.State("LOAD_B_NEXT"):
                with m.If(is_os):
                    m.next = "COMPUTE_OS"
                with m.Else():
                    m.next = "PRELOAD"
            issuer.state(
                "LOAD_B", "LOAD_B_NEXT", CTRL_LOAD, Mux(is_os, DmaOpcode.TRANSPOSE, DmaOpcode.MEMCPY),
                dram_addr=b_tile, local_addr=bufs.b_addr, len=tile_K,
                stride=b_stride, row_bytes=tile_N * in_bytes,
                sync=Mux(is_os, bufs.load(bufs.a_slot, first=False), bufs.load(bufs.b_slot)),
            )  # fmt: skip

            # OS: one COMPUTE per K tile, the K tiles chained into one reduction in
            # the PEs; the last one sends the C tile to the accumulator
            issuer.state(
                "COMPUTE_OS", "NEXT_K", CTRL_EXEC, ExecOpcode.COMPUTE,
                rs1=bufs.a_addr, rs2=bufs.b_addr, rd=bufs.acc_addr, k_dim=tile_K,
                chain=bufs.chain(first_k, last_k),
                sync=bufs.compute(first_k=first_k, last_k=last_k), then=bufs.next_l2(),
            )  # fmt: skip

            # WS: B is stationary and A streams; IS: A (as Aᵀ) is stationary and B
            # streams, so the accumulator holds Cᵀ. Later K tiles accumulate.
            issuer.state(
                "PRELOAD", "COMPUTE", CTRL_EXEC, ExecOpcode.PRELOAD,
                rs1=Mux(df == IS, bufs.a_addr, bufs.b_addr), k_dim=tile_K,
                sync=Mux(df == IS, bufs.preload(bufs.a_slot), bufs.preload(bufs.b_slot)),
            )  # fmt: skip
            issuer.state(
                "COMPUTE", "NEXT_K", CTRL_EXEC,
                Mux(first_k, ExecOpcode.COMPUTE, ExecOpcode.COMPUTE_ACCUMULATE),
                rs1=Mux(df == IS, bufs.b_addr, bufs.a_addr), rd=bufs.acc_addr,
                k_dim=Mux(df == IS, tile_N, tile_M),
                sync=Mux(
                    df == IS,
                    bufs.compute(first_k=first_k, last_k=last_k, slot=bufs.b_slot),
                    bufs.compute(first_k=first_k, last_k=last_k, slot=bufs.a_slot),
                ),
                then=bufs.next_l2(),
            )  # fmt: skip

            with m.State("NEXT_K"):
                with m.If(tile_k + 1 < tiles_K):
                    m.d.sync += tile_k.eq(tile_k + 1)
                    m.next = "LOAD_A"
                with m.Else():
                    m.d.sync += tile_k.eq(0)
                    m.next = "STORE"

            # One STORE per C tile, only its own columns; IS turns Cᵀ back into C
            # (tile_N accumulator rows of tile_M -> tile_M rows of tile_N)
            issuer.state(
                "STORE", "NEXT_IJ", CTRL_STORE,
                Mux(df == IS, DmaOpcode.TRANSPOSE, DmaOpcode.MEMCPY),
                dram_addr=c_tile, local_addr=bufs.acc_addr, len=tile_M,
                stride=c_stride, row_bytes=tile_N * acc_bytes, activation=reg_activation,
                sync=bufs.store(), then=bufs.next_acc(),
            )  # fmt: skip

            with m.State("NEXT_IJ"):
                with m.If(tile_j + 1 < tiles_N):
                    m.d.sync += tile_j.eq(tile_j + 1)
                    m.next = "LOAD_A"
                with m.Elif(tile_i + 1 < tiles_M):
                    m.d.sync += [tile_i.eq(tile_i + 1), tile_j.eq(0)]
                    m.next = "LOAD_A"
                with m.Else():
                    m.next = "DRAIN"

            # Done when the last C tile is in L3
            with m.State("DRAIN"), m.If(issuer.drained()):
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
