"""
Conv2D ISA instruction: Y = conv2d(X, F)

A hardware FSM that runs a 2D convolution on the systolic array as an implicit
GEMM, issuing LOAD / CONFIG / COMPUTE / STORE commands on the command
port ``SystolicTop`` takes (see ``isa.command``).

Tensor layouts (dense, NHWC):
    X (input):  [batch, in_h, in_w, in_c]
    F (filter): [kernel_h, kernel_w, in_c, out_c]
    Y (output): [batch, out_h, out_w, out_c], output positions ``y_stride`` bytes
        apart (0 = dense, ``out_c`` results)

As a matrix multiply: M = batch * out_h * out_w output positions (rows),
N = out_c (columns), and K = kernel_h * kernel_w * in_c, walked as one K tile per
kernel offset (kh, kw) and chunk of ``dim`` input channels. Dataflow:
output-stationary: one LOAD per output position writes that position's channels
as one L2 row, which is the A tile row-major, as OS streams it. (WS and IS stream
A column-major, which per-position LOADs cannot write, so Conv2d needs an array
with OS.) For each output tile (M tile, N tile), and each K tile:

- LOAD one row per output position ``p = (b, oh, ow)``: the input pixel
  ``X[b, oh*stride_h + kh*dilation_h - pad_h, ow*stride_w + kw*dilation_w - pad_w,
  ic0:]`` (its chunk's channels), or, where that falls in the padding, the zero
  row at ``CONFIG_ZERO_ADDR``;
- LOAD the filter tile ``F[kh, kw, ic0:, n0:]`` (one command: rows ``out_c``
  elements apart) with a corner turn, so it lands column-major, and COMPUTE: the
  K tiles are one reduction, held in the PEs (``cmd_chain``), and the last one
  writes the output tile to the accumulator;

then STORE the output tile, one command: rows ``y_stride`` bytes apart, only the
tile's own channels written (byte strobes), so Y can be dense. The tile's
BlockMover moves rows from and to any byte address, so X, F and Y need no alignment.

The commands are pushed without waiting for each other (``isa.command``): credits
and tile-ready tokens order them in ``SystolicTop``. A K tile's X rows and its
filter tile are read by one COMPUTE, so they share one slot (the pair's A slot):
the first X row's LOAD takes its credit and the filter tile's LOAD emits its
tile-ready token. Each K tile uses the next A/B pair and each output tile the next
accumulator slot, in ring order (``TileBuffers``, ``buffers=2`` by default; 1
serializes the processes). The instruction is done when its last STORE completes.

Errors (``error_code``): CONFIG_INCOMPLETE without the input, output and kernel
dimensions and the X/F/Y addresses (or with padding but no zero row);
INVALID_DIMS for a zero dimension, stride or dilation; UNSUPPORTED_BIAS for a
bias and UNSUPPORTED_ACCUMULATE for accumulating onto an existing Y
(``CONFIG_OPTIONS`` bit 0), which both need a load into the accumulator;
UNSUPPORTED_LAYOUT when output positions are closer than ``out_c`` results
(``y_stride``), so rows would overlap.

The caller sets ``out_h``/``out_w`` consistently with the input, kernel, stride,
padding and dilation; they are not derived.
"""

from enum import IntEnum

from amaranth import Module, Mux, Signal, signed
from amaranth.lib.wiring import Component, In, Out

from ..config import Dataflow, SystolicConfig
from ..util.commands import DmaOpcode, ExecOpcode
from .command import CTRL_EXEC, CTRL_LOAD, CTRL_STORE, CommandIssuer, TileBuffers, command_ports


class Conv2dCmd(IntEnum):
    """Configuration commands for the Conv2D instruction."""

    CONFIG_INPUT_DIMS = 0x01  # data[7:0]=batch, [23:8]=height, [39:24]=width, [55:40]=channels_in
    CONFIG_OUTPUT_DIMS = 0x02  # data[15:0]=out_height, [31:16]=out_width, [47:32]=channels_out
    CONFIG_KERNEL_DIMS = 0x03  # data[7:0]=kernel_h, [15:8]=kernel_w
    CONFIG_STRIDE = 0x04  # data[7:0]=stride_h, [15:8]=stride_w
    CONFIG_PADDING = 0x05  # data[7:0]=pad_h, [15:8]=pad_w
    CONFIG_DILATION = 0x06  # data[7:0]=dilation_h, [15:8]=dilation_w
    CONFIG_X_ADDR = 0x10
    CONFIG_F_ADDR = 0x11
    CONFIG_Y_ADDR = 0x12
    CONFIG_B_ADDR = 0x13  # Bias address, 0 = no bias (bias is not supported)
    CONFIG_ZERO_ADDR = 0x14  # A zero-filled DRAM row, loaded for padded positions
    CONFIG_X_STRIDE = 0x20  # Reserved (X is dense NHWC)
    CONFIG_F_STRIDE = 0x21  # Reserved (F is dense)
    CONFIG_Y_STRIDE = 0x22  # Bytes between output positions (0 = dense)
    CONFIG_B_STRIDE = 0x23  # Reserved
    CONFIG_OPTIONS = 0x30  # data[0]=accumulate, data[3:1]=activation
    START = 0xF0
    ABORT = 0xFF


class Conv2dError(IntEnum):
    """Error codes for the Conv2D instruction."""

    NONE = 0x00
    INVALID_DIMS = 0x01  # A dimension, stride or dilation is zero
    CONFIG_INCOMPLETE = 0x05  # Dimensions or addresses missing (or padding without a zero row)
    UNSUPPORTED_LAYOUT = 0x07  # See the module docstring
    UNSUPPORTED_BIAS = 0x08  # A bias (B) was configured
    UNSUPPORTED_ACCUMULATE = 0x09  # Accumulate onto Y (Y is not loaded into the accumulator)


OS = 0  # ExecuteController CONFIG rs1

# batch (8 bits) * out_h (16) * out_w (16): the widest M, and M-tile count, configurable
M_BITS = 40


class Conv2d(Component):
    """
    Conv2D ISA instruction: Y = conv2d(X, F), as an implicit GEMM.

    Ports:
        cfg_valid, cfg_ready, cfg_cmd, cfg_data: Configuration (``Conv2dCmd``)
        cmd_*: Command port (``isa.command``)
        busy, done, error, error_code: Status
        progress_m, progress_n, progress_k: Current M tile, N tile, K tile
        selected_dataflow: 0 (output-stationary)
        param_*: Configured and derived dimensions
        dbg_addr_X/F/Y/B, dbg_tile_M/N/K: Current X row, F and Y tile addresses; tile sizes
    """

    def __init__(self, config: SystolicConfig, buffers: int = 2):
        self.config = config
        rows = config.grid_rows * config.tile_rows
        cols = config.grid_cols * config.tile_cols
        if rows != cols:
            raise ValueError(f"Conv2d requires a square array, got {rows}x{cols}")
        self.dim = rows
        if self.dim & (self.dim - 1):
            raise ValueError(f"Conv2d requires a power-of-two array dimension, got {self.dim}")
        if not config.dataflow & Dataflow.OUTPUT_STATIONARY:
            raise ValueError(
                "Conv2d issues output-stationary commands: the array must support OS "
                "(its per-position LOADs write A row-major)"
            )
        # X tiles in the A slots, filter tiles in the B slots, Y tiles in the accumulator slots
        self.tiles = TileBuffers(config, self.dim, buffers, "Conv2d")

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
            "progress_m": Out(M_BITS),
            "progress_n": Out(16),
            "progress_k": Out(16),
            "selected_dataflow": Out(2),
            "param_batch": Out(8),
            "param_in_h": Out(16),
            "param_in_w": Out(16),
            "param_in_c": Out(16),
            "param_out_h": Out(16),
            "param_out_w": Out(16),
            "param_out_c": Out(16),
            "param_kernel_h": Out(8),
            "param_kernel_w": Out(8),
            "param_M": Out(M_BITS),
            "param_N": Out(32),
            "param_K": Out(32),
            "dbg_addr_X": Out(64),
            "dbg_addr_F": Out(64),
            "dbg_addr_Y": Out(64),
            "dbg_addr_B": Out(64),
            "dbg_tile_M": Out(16),
            "dbg_tile_N": Out(16),
            "dbg_tile_K": Out(16),
            "dbg_tiles_M": Out(M_BITS),
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

        # Configuration
        batch, kernel_h, kernel_w = Signal(8), Signal(8), Signal(8)
        in_h, in_w, in_c = Signal(16), Signal(16), Signal(16)
        out_h, out_w, out_c = Signal(16), Signal(16), Signal(16)
        stride_h, stride_w = Signal(8, init=1), Signal(8, init=1)
        pad_h, pad_w = Signal(8), Signal(8)
        dil_h, dil_w = Signal(8, init=1), Signal(8, init=1)
        x_addr, f_addr, y_addr, b_addr, zero_addr = (
            Signal(64, name=f"reg_{n}_addr") for n in "XFYBZ"
        )
        y_stride_cfg = Signal(32)
        accumulate = Signal()
        activation = Signal(3)
        in_valid, out_valid, kernel_valid, addrs_valid, zero_valid = (Signal() for _ in range(5))

        # Derived sizes
        # M = batch * out_h * out_w needs M_BITS; the M tile counters are as wide
        reg_M, reg_N, reg_K = Signal(M_BITS), Signal(32), Signal(32)
        tiles_M, tiles_N, tiles_C = (
            Signal(M_BITS),
            Signal(16),
            Signal(16),
        )  # C: input-channel chunks
        y_stride = Signal(32)  # bytes between output positions

        # Loop state
        tile_m, tile_n = Signal(M_BITS), Signal(16)
        kh, kw, chunk = Signal(8), Signal(8), Signal(16)
        row = Signal(16)
        # Position (b, oh, ow) of the M tile's first row, of the current row, and
        # of the row after the tile (the next M tile's first)
        base_b, base_oh, base_ow = Signal(8), Signal(16), Signal(16)
        pos_b, pos_oh, pos_ow = Signal(8), Signal(16), Signal(16)
        next_b, next_oh, next_ow = Signal(8), Signal(16), Signal(16)

        m0 = Signal(M_BITS + 8)
        n0 = Signal(32)
        ic0 = Signal(32)
        m.d.comb += [
            m0.eq(tile_m << dim_shift),
            n0.eq(tile_n << dim_shift),
            ic0.eq(chunk << dim_shift),
        ]

        tile_M, tile_N, tile_K = Signal(16), Signal(16), Signal(16)
        for size, total, start in ((tile_M, reg_M, m0), (tile_N, reg_N, n0), (tile_K, in_c, ic0)):
            remaining = Signal(M_BITS + 8)
            m.d.comb += [
                remaining.eq(total - start),
                size.eq(Mux(remaining < dim, remaining[:16], dim)),
            ]

        # Input pixel of the current row at kernel offset (kh, kw)
        ih, iw = Signal(signed(32)), Signal(signed(32))
        in_bounds = Signal()
        m.d.comb += [
            ih.eq(pos_oh * stride_h + kh * dil_h - pad_h),
            iw.eq(pos_ow * stride_w + kw * dil_w - pad_w),
            in_bounds.eq((ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w)),
        ]
        x_row, f_tile, y_tile = Signal(64), Signal(64), Signal(64)
        pixel = Signal(64)
        m.d.comb += [
            pixel.eq((pos_b * in_h + ih.as_unsigned()) * in_w + iw.as_unsigned()),
            x_row.eq(Mux(in_bounds, x_addr + (pixel * in_c + ic0) * in_bytes, zero_addr)),
            f_tile.eq(f_addr + (((kh * kernel_w + kw) * in_c + ic0) * out_c + n0) * in_bytes),
            y_tile.eq(y_addr + m0 * y_stride + n0 * acc_bytes),
        ]

        # Output rows must not overlap (see the module docstring)
        y_stride_ok = Signal()
        m.d.comb += y_stride_ok.eq(y_stride >= out_c * acc_bytes)

        m.d.comb += [
            self.param_batch.eq(batch),
            self.param_in_h.eq(in_h),
            self.param_in_w.eq(in_w),
            self.param_in_c.eq(in_c),
            self.param_out_h.eq(out_h),
            self.param_out_w.eq(out_w),
            self.param_out_c.eq(out_c),
            self.param_kernel_h.eq(kernel_h),
            self.param_kernel_w.eq(kernel_w),
            self.param_M.eq(reg_M),
            self.param_N.eq(reg_N),
            self.param_K.eq(reg_K),
            self.selected_dataflow.eq(OS),
            self.progress_m.eq(tile_m),
            self.progress_n.eq(tile_n),
            self.progress_k.eq((kh * kernel_w + kw) * tiles_C + chunk),
            self.dbg_addr_X.eq(x_row),
            self.dbg_addr_F.eq(f_tile),
            self.dbg_addr_Y.eq(y_tile),
            self.dbg_addr_B.eq(b_addr),
            self.dbg_tile_M.eq(tile_M),
            self.dbg_tile_N.eq(tile_N),
            self.dbg_tile_K.eq(tile_K),
            self.dbg_tiles_M.eq(tiles_M),
        ]

        # The position after the current row's (b, oh, ow), in NHWC order
        succ_b, succ_oh, succ_ow = Signal(8), Signal(16), Signal(16)
        last_ow = pos_ow + 1 == out_w
        last_oh = pos_oh + 1 == out_h
        m.d.comb += [
            succ_ow.eq(Mux(last_ow, 0, pos_ow + 1)),
            succ_oh.eq(Mux(last_ow, Mux(last_oh, 0, pos_oh + 1), pos_oh)),
            succ_b.eq(Mux(last_ow & last_oh, pos_b + 1, pos_b)),
        ]

        first_k = (kh == 0) & (kw == 0) & (chunk == 0)
        last_k = (chunk + 1 >= tiles_C) & (kw + 1 >= kernel_w) & (kh + 1 >= kernel_h)
        bufs = self.tiles.build(m)

        with m.FSM(init="IDLE") as fsm:
            with m.State("IDLE"):
                m.d.comb += self.cfg_ready.eq(1)
                d = self.cfg_data
                with m.If(self.cfg_valid), m.Switch(self.cfg_cmd):
                    with m.Case(Conv2dCmd.CONFIG_INPUT_DIMS):
                        m.d.sync += [
                            batch.eq(d[0:8]), in_h.eq(d[8:24]), in_w.eq(d[24:40]),
                            in_c.eq(d[40:56]), in_valid.eq(1),
                        ]  # fmt: skip
                    with m.Case(Conv2dCmd.CONFIG_OUTPUT_DIMS):
                        m.d.sync += [
                            out_h.eq(d[0:16]), out_w.eq(d[16:32]), out_c.eq(d[32:48]),
                            out_valid.eq(1),
                        ]  # fmt: skip
                    with m.Case(Conv2dCmd.CONFIG_KERNEL_DIMS):
                        m.d.sync += [kernel_h.eq(d[0:8]), kernel_w.eq(d[8:16]), kernel_valid.eq(1)]
                    with m.Case(Conv2dCmd.CONFIG_STRIDE):
                        m.d.sync += [stride_h.eq(d[0:8]), stride_w.eq(d[8:16])]
                    with m.Case(Conv2dCmd.CONFIG_PADDING):
                        m.d.sync += [pad_h.eq(d[0:8]), pad_w.eq(d[8:16])]
                    with m.Case(Conv2dCmd.CONFIG_DILATION):
                        m.d.sync += [dil_h.eq(d[0:8]), dil_w.eq(d[8:16])]
                    with m.Case(Conv2dCmd.CONFIG_X_ADDR):
                        m.d.sync += x_addr.eq(d)
                    with m.Case(Conv2dCmd.CONFIG_F_ADDR):
                        m.d.sync += f_addr.eq(d)
                    with m.Case(Conv2dCmd.CONFIG_Y_ADDR):
                        m.d.sync += [y_addr.eq(d), addrs_valid.eq(1)]
                    with m.Case(Conv2dCmd.CONFIG_B_ADDR):
                        m.d.sync += b_addr.eq(d)
                    with m.Case(Conv2dCmd.CONFIG_ZERO_ADDR):
                        m.d.sync += [zero_addr.eq(d), zero_valid.eq(1)]
                    with m.Case(Conv2dCmd.CONFIG_Y_STRIDE):
                        m.d.sync += y_stride_cfg.eq(d[0:32])
                    with m.Case(Conv2dCmd.CONFIG_OPTIONS):
                        m.d.sync += [accumulate.eq(d[0]), activation.eq(d[1:4])]
                    with m.Case(Conv2dCmd.START):
                        with m.If(in_valid & out_valid & kernel_valid & addrs_valid):
                            m.d.sync += [
                                reg_M.eq(batch * out_h * out_w),
                                reg_N.eq(out_c),
                                reg_K.eq(kernel_h * kernel_w * in_c),
                                y_stride.eq(Mux(y_stride_cfg == 0, out_c * acc_bytes, y_stride_cfg)),
                                tile_m.eq(0), tile_n.eq(0), kh.eq(0), kw.eq(0), chunk.eq(0), row.eq(0),
                                base_b.eq(0), base_oh.eq(0), base_ow.eq(0),
                                *issuer.reset(),
                                *bufs.reset(),
                            ]  # fmt: skip
                            m.next = "TILES"
                        with m.Else():
                            m.d.sync += self.error_code.eq(Conv2dError.CONFIG_INCOMPLETE)
                            m.next = "ERROR"

            with m.State("TILES"):
                m.d.sync += [
                    tiles_M.eq((reg_M + dim - 1) >> dim_shift),
                    tiles_N.eq((reg_N + dim - 1) >> dim_shift),
                    tiles_C.eq((in_c + dim - 1) >> dim_shift),
                ]
                m.next = "VALIDATE"

            with m.State("VALIDATE"):
                zero_dim = Signal()
                m.d.comb += zero_dim.eq(
                    (batch == 0) | (in_h == 0) | (in_w == 0) | (in_c == 0)
                    | (out_h == 0) | (out_w == 0) | (out_c == 0)
                    | (kernel_h == 0) | (kernel_w == 0)
                    | (stride_h == 0) | (stride_w == 0) | (dil_h == 0) | (dil_w == 0)
                )  # fmt: skip
                with m.If(zero_dim):
                    m.d.sync += self.error_code.eq(Conv2dError.INVALID_DIMS)
                    m.next = "ERROR"
                with m.Elif(((pad_h != 0) | (pad_w != 0)) & ~zero_valid):
                    m.d.sync += self.error_code.eq(Conv2dError.CONFIG_INCOMPLETE)
                    m.next = "ERROR"
                with m.Elif(b_addr != 0):
                    m.d.sync += self.error_code.eq(Conv2dError.UNSUPPORTED_BIAS)
                    m.next = "ERROR"
                with m.Elif(accumulate):
                    m.d.sync += self.error_code.eq(Conv2dError.UNSUPPORTED_ACCUMULATE)
                    m.next = "ERROR"
                with m.Elif(~y_stride_ok):
                    m.d.sync += self.error_code.eq(Conv2dError.UNSUPPORTED_LAYOUT)
                    m.next = "ERROR"
                with m.Else():
                    m.d.sync += self.error_code.eq(Conv2dError.NONE)
                    m.next = "CONFIG"

            issuer.state("CONFIG", "K_TILE", CTRL_EXEC, ExecOpcode.CONFIG, rs1=OS)

            # Start a K tile: walk the M tile's positions from its first
            with m.State("K_TILE"):
                m.d.sync += [row.eq(0), pos_b.eq(base_b), pos_oh.eq(base_oh), pos_ow.eq(base_ow)]
                m.next = "LOAD_X"

            issuer.state(
                "LOAD_X", "X_NEXT", CTRL_LOAD, DmaOpcode.MEMCPY,
                dram_addr=x_row, local_addr=bufs.a_addr + row, len=1, row_bytes=tile_K * in_bytes,
                sync=bufs.load(bufs.a_slot, first=row == 0, last=False),
            )  # fmt: skip
            with m.State("X_NEXT"):
                m.d.sync += [pos_b.eq(succ_b), pos_oh.eq(succ_oh), pos_ow.eq(succ_ow)]
                with m.If(row + 1 < tile_M):
                    m.d.sync += row.eq(row + 1)
                    m.next = "LOAD_X"
                with m.Else():
                    # The position after the tile's last row starts the next M tile
                    m.d.sync += [
                        row.eq(0),
                        next_b.eq(succ_b),
                        next_oh.eq(succ_oh),
                        next_ow.eq(succ_ow),
                    ]
                    m.next = "LOAD_F"

            # The filter tile, one command: rows out_c elements apart, turned so
            # that L2 row n is output channel n's filter column. It completes the
            # pair, so it emits the A slot's token.
            issuer.state(
                "LOAD_F", "COMPUTE", CTRL_LOAD, DmaOpcode.TRANSPOSE,
                dram_addr=f_tile, local_addr=bufs.b_addr, len=tile_K,
                sync=bufs.load(bufs.a_slot, first=False),
                stride=out_c * in_bytes, row_bytes=tile_N * in_bytes,
            )  # fmt: skip

            # One COMPUTE per K tile, the K tiles one reduction in the PEs
            issuer.state(
                "COMPUTE", "NEXT_K", CTRL_EXEC, ExecOpcode.COMPUTE,
                rs1=bufs.a_addr, rs2=bufs.b_addr, rd=bufs.acc_addr, k_dim=tile_K,
                chain=bufs.chain(first_k, last_k),
                sync=bufs.compute(first_k=first_k, last_k=last_k), then=bufs.next_l2(),
            )  # fmt: skip

            # K tiles: input-channel chunk, then kernel column, then kernel row
            with m.State("NEXT_K"):
                with m.If(chunk + 1 < tiles_C):
                    m.d.sync += chunk.eq(chunk + 1)
                    m.next = "K_TILE"
                with m.Elif(kw + 1 < kernel_w):
                    m.d.sync += [chunk.eq(0), kw.eq(kw + 1)]
                    m.next = "K_TILE"
                with m.Elif(kh + 1 < kernel_h):
                    m.d.sync += [chunk.eq(0), kw.eq(0), kh.eq(kh + 1)]
                    m.next = "K_TILE"
                with m.Else():
                    m.d.sync += [chunk.eq(0), kw.eq(0), kh.eq(0), row.eq(0)]
                    m.next = "STORE"

            # The output tile, one command: only its own channels
            issuer.state(
                "STORE", "NEXT_MN", CTRL_STORE, DmaOpcode.MEMCPY,
                dram_addr=y_tile, local_addr=bufs.acc_addr, len=tile_M,
                stride=y_stride, row_bytes=tile_N * acc_bytes, activation=activation,
                sync=bufs.store(), then=bufs.next_acc(),
            )  # fmt: skip

            with m.State("NEXT_MN"):
                with m.If(tile_n + 1 < tiles_N):
                    m.d.sync += tile_n.eq(tile_n + 1)
                    m.next = "K_TILE"
                with m.Elif(tile_m + 1 < tiles_M):
                    m.d.sync += [
                        tile_m.eq(tile_m + 1), tile_n.eq(0),
                        base_b.eq(next_b), base_oh.eq(next_oh), base_ow.eq(next_ow),
                    ]  # fmt: skip
                    m.next = "K_TILE"
                with m.Else():
                    m.next = "DRAIN"

            # Done when the last output tile is in L3
            with m.State("DRAIN"), m.If(issuer.drained()):
                m.next = "DONE"

            with m.State("DONE"):
                m.d.comb += self.done.eq(1)
                m.d.sync += [
                    in_valid.eq(0), out_valid.eq(0), kernel_valid.eq(0), addrs_valid.eq(0), zero_valid.eq(0),
                ]  # fmt: skip
                m.next = "IDLE"

            with m.State("ERROR"):
                m.d.comb += self.error.eq(1)
                m.next = "IDLE"

        m.d.comb += self.busy.eq(~fsm.ongoing("IDLE"))
        return m
