"""
Unit tests for the Conv2D ISA instruction (an implicit GEMM).

The command stream Conv2d issues is run through a Python model of SystolicTop
(``_isa.TopModel``) and the output compared with a numpy NHWC convolution: this
checks the address arithmetic (output-position decoding, kernel offsets, stride,
padding, dilation, channel chunks and N tiles). The layout check is off for
those, because the tile's DMA cannot load these layouts (see ``isa/conv2d.py``).
"""

import numpy as np
import pytest

from systars.config import Dataflow, SystolicConfig
from systars.isa.command import CTRL_EXEC, CTRL_LOAD, CTRL_STORE
from systars.isa.conv2d import Conv2d, Conv2dCmd, Conv2dError
from systars.util.commands import ExecOpcode

from ._isa import TopModel, run_isa, to_unsigned_bytes
from ._sim import skip_without_yosys

X_ADDR, F_ADDR, Y_ADDR, ZERO_ADDR = 0x1000, 0x20000, 0x40000, 0xF0000
DIM = 4


def config(**overrides):
    fields = {"grid_rows": DIM, "grid_cols": DIM, "sp_capacity_kb": 4, "acc_capacity_kb": 4}
    fields.update(overrides)
    return SystolicConfig(**fields)


def out_size(size, kernel, stride, pad, dilation):
    return (size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1


def reference(X, F, stride, pad, dilation):
    batch, in_h, in_w, _ = X.shape
    kh_n, kw_n, _, out_c = F.shape
    oh_n = out_size(in_h, kh_n, stride[0], pad[0], dilation[0])
    ow_n = out_size(in_w, kw_n, stride[1], pad[1], dilation[1])
    Y = np.zeros((batch, oh_n, ow_n, out_c), dtype=np.int64)
    for b in range(batch):
        for oh in range(oh_n):
            for ow in range(ow_n):
                for kh in range(kh_n):
                    for kw in range(kw_n):
                        ih = oh * stride[0] + kh * dilation[0] - pad[0]
                        iw = ow * stride[1] + kw * dilation[1] - pad[1]
                        if 0 <= ih < in_h and 0 <= iw < in_w:
                            Y[b, oh, ow] += X[b, ih, iw] @ F[kh, kw]
    return Y


def setup(
    X_shape,
    F_shape,
    stride=(1, 1),
    pad=(0, 0),
    dilation=(1, 1),
    *,
    y_stride=0,
    zero=True,
    bias=0,
    options=0,
):
    batch, in_h, in_w, in_c = X_shape
    kh, kw, _, out_c = F_shape
    oh = out_size(in_h, kh, stride[0], pad[0], dilation[0])
    ow = out_size(in_w, kw, stride[1], pad[1], dilation[1])
    cmds = [
        (Conv2dCmd.CONFIG_INPUT_DIMS, batch | in_h << 8 | in_w << 24 | in_c << 40),
        (Conv2dCmd.CONFIG_OUTPUT_DIMS, oh | ow << 16 | out_c << 32),
        (Conv2dCmd.CONFIG_KERNEL_DIMS, kh | kw << 8),
        (Conv2dCmd.CONFIG_STRIDE, stride[0] | stride[1] << 8),
        (Conv2dCmd.CONFIG_PADDING, pad[0] | pad[1] << 8),
        (Conv2dCmd.CONFIG_DILATION, dilation[0] | dilation[1] << 8),
        (Conv2dCmd.CONFIG_X_ADDR, X_ADDR),
        (Conv2dCmd.CONFIG_F_ADDR, F_ADDR),
        (Conv2dCmd.CONFIG_Y_ADDR, Y_ADDR),
        (Conv2dCmd.CONFIG_B_ADDR, bias),
        (Conv2dCmd.CONFIG_Y_STRIDE, y_stride),
        (Conv2dCmd.CONFIG_OPTIONS, options),
    ]
    if zero:
        cmds.append((Conv2dCmd.CONFIG_ZERO_ADDR, ZERO_ADDR))
    return cmds


def y_stride_for(out_c):
    """Bytes between output positions: whole N tiles of 32-bit results."""
    return -(-out_c // DIM) * DIM * 4


def check(X_shape, F_shape, stride=(1, 1), pad=(0, 0), dilation=(1, 1), seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(-128, 128, X_shape)
    F = rng.integers(-128, 128, F_shape)
    expected = reference(X, F, stride, pad, dilation)
    y_stride = y_stride_for(F_shape[3])

    cfg = config()
    isa = Conv2d(cfg, require_beat_aligned=False)
    result = run_isa(
        isa,
        setup(X_shape, F_shape, stride, pad, dilation, y_stride=y_stride),
        Conv2dCmd.START,
        latency=0,
    )
    assert result.done and not result.error, f"error {result.error_code:#x}"

    model = TopModel(cfg)
    model.write(X_ADDR, to_unsigned_bytes(X, 8))
    model.write(F_ADDR, to_unsigned_bytes(F, 8))
    model.run(result.commands)
    positions = expected.shape[0] * expected.shape[1] * expected.shape[2]
    out_c = F_shape[3]
    got = np.array([model.read_elems(Y_ADDR + p * y_stride, 32, out_c) for p in range(positions)])
    np.testing.assert_array_equal(got, expected.reshape(positions, out_c))
    return result


class TestNumerical:
    def test_pointwise(self):
        check((1, 3, 3, 3), (1, 1, 3, 4))

    def test_3x3_same_padding_two_channel_chunks_two_n_tiles_two_batches(self):
        """M tiles cross output rows and batches; K walks 9 offsets x 2 channel chunks."""
        check((2, 5, 5, 6), (3, 3, 6, 6), pad=(1, 1), seed=1)

    def test_stride_dilation_rectangular_kernel(self):
        check((1, 9, 7, 2), (2, 3, 2, 3), stride=(2, 1), pad=(2, 1), dilation=(2, 2), seed=2)

    def test_stride_two_no_padding(self):
        check((1, 8, 8, 4), (3, 3, 4, 2), stride=(2, 2), seed=3)

    def test_kernel_larger_than_padding_reach(self):
        """Output positions whose whole window is padding read only zero rows."""
        check((1, 2, 2, 1), (3, 3, 1, 1), pad=(3, 3), seed=4)


class TestCommandStream:
    def commands(self, **kw):
        X_shape, F_shape = (1, 4, 4, 2), (3, 3, 2, 3)
        cfg = config()
        cmds = setup(X_shape, F_shape, pad=(1, 1), y_stride=y_stride_for(3), **kw)
        return run_isa(
            Conv2d(cfg, require_beat_aligned=False), cmds, Conv2dCmd.START, latency=0
        ).commands

    def test_structure(self):
        cmds = self.commands()
        assert (cmds[0].type, cmds[0].opcode, cmds[0].rs1) == (CTRL_EXEC, ExecOpcode.CONFIG, 1)
        execs = [c.opcode for c in cmds[1:] if c.type == CTRL_EXEC]
        # 16 positions = 4 M tiles, 1 N tile, 9 K tiles (3x3 kernel, 1 channel chunk)
        per_m_tile = [ExecOpcode.PRELOAD, ExecOpcode.COMPUTE] + [
            ExecOpcode.PRELOAD,
            ExecOpcode.COMPUTE_ACCUMULATE,
        ] * 8
        assert execs == per_m_tile * 4
        assert len([c for c in cmds if c.type == CTRL_STORE]) == 16
        assert all(c.len == 1 for c in cmds if c.type in (CTRL_LOAD, CTRL_STORE))

    def test_padding_rows_load_the_zero_row(self):
        cmds = self.commands()
        x_loads = [c for c in cmds if c.type == CTRL_LOAD and c.local_addr < DIM]
        zero = [c for c in x_loads if c.dram_addr == ZERO_ADDR]
        # Kernel offset (0, 0) of the first M tile (positions (0,0)..(0,3)) is all padding
        assert all(c.dram_addr == ZERO_ADDR for c in x_loads[:4])
        assert zero and len(zero) < len(x_loads)


class TestErrors:
    def run(self, cmds, aligned=False):
        return run_isa(Conv2d(config(), require_beat_aligned=aligned), cmds, Conv2dCmd.START)

    def test_start_without_config(self):
        assert self.run([]).error_code == Conv2dError.CONFIG_INCOMPLETE

    def test_padding_needs_zero_row(self):
        cmds = setup((1, 4, 4, 4), (3, 3, 4, 4), pad=(1, 1), zero=False)
        assert self.run(cmds).error_code == Conv2dError.CONFIG_INCOMPLETE

    def test_zero_stride(self):
        cmds = setup((1, 4, 4, 4), (1, 1, 4, 4))
        cmds.append((Conv2dCmd.CONFIG_STRIDE, 0))
        assert self.run(cmds).error_code == Conv2dError.INVALID_DIMS

    def test_bias_is_unsupported(self):
        cmds = setup((1, 4, 4, 4), (1, 1, 4, 4), bias=0x8000)
        assert self.run(cmds).error_code == Conv2dError.UNSUPPORTED_BIAS

    def test_accumulate_is_unsupported(self):
        """Y is never loaded into the accumulator, so accumulating onto it would add stale rows."""
        cmds = setup((1, 4, 4, 4), (1, 1, 4, 4), y_stride=y_stride_for(4), options=1)
        result = self.run(cmds)
        assert result.error_code == Conv2dError.UNSUPPORTED_ACCUMULATE
        assert not result.commands

    def test_dense_output_with_partial_n_tile(self):
        """out_c = 5 needs 8 results per position (two N tiles); dense is 5: rejected."""
        cmds = setup((1, 4, 4, 4), (1, 1, 4, 5))
        result = self.run(cmds)
        assert result.error_code == Conv2dError.UNSUPPORTED_LAYOUT
        assert not result.commands

    def test_default_rejects_layouts_the_dma_cannot_move(self):
        cmds = setup((1, 4, 4, 4), (1, 1, 4, 4))
        assert self.run(cmds, aligned=True).error_code == Conv2dError.UNSUPPORTED_LAYOUT


class TestConstruction:
    def test_requires_weight_stationary(self):
        with pytest.raises(ValueError, match="weight-stationary"):
            Conv2d(config(dataflow=Dataflow.OUTPUT_STATIONARY))

    def test_requires_square_power_of_two(self):
        with pytest.raises(ValueError, match="square"):
            Conv2d(config(grid_cols=2))
        with pytest.raises(ValueError, match="power-of-two"):
            Conv2d(config(grid_rows=3, grid_cols=3))

    def test_verilog(self):
        skip_without_yosys()
        from amaranth.back import verilog

        assert "module Conv2d" in verilog.convert(Conv2d(config()), name="Conv2d")


def test_m_tile_count_does_not_truncate():
    """batch 8 x 224 x 224 = 401,408 positions = 100,352 M tiles: wider than 16 bits."""
    from ._sim import run_sim

    isa = Conv2d(config(), require_beat_aligned=False)
    cmds = setup((8, 224, 224, 4), (1, 1, 4, 4), y_stride=y_stride_for(4))
    seen = {}

    async def testbench(ctx):
        for cmd, data in [*cmds, (Conv2dCmd.START, 0)]:
            ctx.set(isa.cfg_valid, 1)
            ctx.set(isa.cfg_cmd, cmd)
            ctx.set(isa.cfg_data, data)
            await ctx.tick()
        ctx.set(isa.cfg_valid, 0)
        # The first command (CONFIG) is issued after the tile counts are computed
        await ctx.tick().until(isa.cmd_valid)
        seen.update(
            M=ctx.get(isa.param_M), tiles=ctx.get(isa.dbg_tiles_M), error=ctx.get(isa.error)
        )

    run_sim(isa, testbench, max_cycles=200)
    assert seen == {"M": 8 * 224 * 224, "tiles": 8 * 224 * 224 // DIM, "error": 0}
