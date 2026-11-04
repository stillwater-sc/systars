"""
Unit tests for ExecuteController (the Streamer), wired to a real SystolicArray.
Python models serve the scratchpad's two read ports and record the accumulator
writes. Commands are untracked (one at a time, ``cmd_sync`` 0) or tracked (pushed
back to back, ``cmd_sync`` != 0), where the Streamer overlaps them.
"""

import numpy as np
import pytest
from amaranth import Elaboratable, Module

from systars.config import Dataflow, SystolicConfig
from systars.controller.execute import (
    CHAIN_CONTINUES,
    CHAIN_MORE,
    ExecuteController,
)
from systars.core.systolic_array import SystolicArray
from systars.util.commands import ExecOpcode

from ._sim import run_sim, skip_without_yosys

SP_LATENCY = 3
OS, WS, IS = 0, 1, 2
TRACKED = 0x8000  # a cmd_sync value the Streamer passes through untouched (bit 15 unused)


def config(**overrides):
    fields = {"grid_rows": 2, "grid_cols": 2, "tile_rows": 1, "tile_cols": 1}
    fields.update(overrides)
    return SystolicConfig(**fields)


class Harness(Elaboratable):
    def __init__(self, cfg):
        self.cfg = cfg
        self.ctrl = ExecuteController(cfg)
        self.array = SystolicArray(cfg)
        self.rows, self.cols, self.kb = self.ctrl.rows, self.ctrl.cols, self.ctrl.block

    def elaborate(self, _platform):
        m = Module()
        m.submodules.ctrl = ctrl = self.ctrl
        m.submodules.array = array = self.array
        m.d.comb += [
            array.in_shift.eq(ctrl.array_in_shift),
            array.in_control_dataflow.eq(ctrl.array_in_dataflow),
            ctrl.array_out_collision.eq(array.out_collision),
        ]
        for i in range(self.rows):
            for name in ("a", "valid", "first", "last"):
                m.d.comb += getattr(array, f"in_{name}_{i}").eq(
                    getattr(ctrl, f"array_in_{name}_{i}")
                )
        for j in range(self.cols):
            m.d.comb += [
                getattr(array, f"in_b_{j}").eq(getattr(ctrl, f"array_in_b_{j}")),
                getattr(array, f"in_w_{j}").eq(getattr(ctrl, f"array_in_w_{j}")),
            ]
            for name in ("c", "c_valid", "psum", "psum_valid"):
                m.d.comb += getattr(ctrl, f"array_out_{name}_{j}").eq(
                    getattr(array, f"out_{name}_{j}")
                )
        return m


def pack(values, bits):
    return sum((int(v) & ((1 << bits) - 1)) << (i * bits) for i, v in enumerate(values))


def unpack(word, bits, n):
    vals = [(word >> (i * bits)) & ((1 << bits) - 1) for i in range(n)]
    return [v - (1 << bits) if v >> (bits - 1) else v for v in vals]


def run(harness, scratchpad, commands, *, tracked=False, max_cycles=4000):
    """Issue ``commands`` (opcode, fields) and log reads, writes, events and streams.

    Untracked: one command at a time, each waited for. Tracked: pushed back to back
    (``cmd_ready`` only), then the run waits for every completion and result.
    """
    ctrl, cfg = harness.ctrl, harness.cfg
    log = {
        "a_reads": [], "b_reads": [], "writes": [], "completed": [], "results": [],
        "dataflow": [], "streaming": [], "accepted": [], "error": 0,
    }  # fmt: skip

    def port_model(port):
        async def process(ctx):
            pipe = [None] * SP_LATENCY
            async for _clk, _, req, addr in ctx.tick().sample(
                getattr(ctrl, f"sp_{port}_read_req"), getattr(ctrl, f"sp_{port}_read_addr")
            ):
                if req:
                    log[f"{port}_reads"].append(addr)
                pipe = [addr if req else None, *pipe[:-1]]
                out = pipe[-1]
                ctx.set(getattr(ctrl, f"sp_{port}_read_valid"), out is not None)
                ctx.set(getattr(ctrl, f"sp_{port}_read_data"), scratchpad.get(out, 0))

        return process

    async def monitor(ctx):
        t = 0
        async for sample in ctx.tick().sample(
            ctrl.acc_write_req, ctrl.acc_write_addr, ctrl.acc_write_data, ctrl.acc_accumulate,
            ctrl.completed, ctrl.completed_id, ctrl.completed_sync,
            ctrl.result, ctrl.result_id, ctrl.result_sync,
            ctrl.array_in_valid_0, ctrl.error,
        ):  # fmt: skip
            _, _, req, addr, data, acc, done, done_id, done_sync, res, res_id, res_sync = sample[
                :12
            ]
            streaming, error = sample[12], sample[13]
            if req:
                log["writes"].append((addr, unpack(data, cfg.acc_bits, harness.cols), acc))
            if done:
                log["completed"].append((t, done_id, done_sync))
            if res:
                log["results"].append((t, res_id, res_sync))
            log["streaming"].append(streaming)
            log["error"] |= error
            t += 1

    producing = 0

    async def testbench(ctx):
        nonlocal producing
        for n, (opcode, fields) in enumerate(commands):
            ctx.set(ctrl.cmd_opcode, opcode)
            ctx.set(ctrl.cmd_id, n + 1)
            for name in ("rs1", "rs2", "rd", "k_dim", "chain"):
                ctx.set(getattr(ctrl, f"cmd_{name}"), fields.get(name, 0))
            ctx.set(ctrl.cmd_sync, TRACKED | n if tracked else 0)
            ctx.set(ctrl.cmd_valid, 1)
            await ctx.tick().until(ctrl.cmd_ready)
            ctx.set(ctrl.cmd_valid, 0)
            if not tracked:
                await ctx.tick().until(ctrl.completed)
                log["dataflow"].append(ctx.get(ctrl.cfg_dataflow))
        while ctx.get(ctrl.busy) or len(log["completed"]) < len(commands):
            await ctx.tick()
        await ctx.tick().repeat(2)

    run_sim(
        harness,
        testbench,
        processes=[port_model("a"), port_model("b"), monitor],
        max_cycles=max_cycles,
    )
    assert log["error"] == 0
    return log


def written(log):
    """The accumulator, as written: address -> row (the last write wins)."""
    return {addr: row for addr, row, _ in log["writes"]}


def os_tiles(harness, A, B, a_base, b_base):
    """Lay out ``A @ B`` (OS) in L2 as K tiles; returns (scratchpad, commands)."""
    R, C, KB = harness.rows, harness.cols, harness.kb
    cfg, sp, cmds = harness.cfg, {}, []
    k_total = A.shape[1]
    starts = list(range(0, k_total, KB))
    for t, k0 in enumerate(starts):
        kn = min(KB, k_total - k0)
        for i in range(R):
            sp[a_base + t * R + i] = pack(A[i, k0 : k0 + kn], cfg.input_bits)
        for j in range(C):
            sp[b_base + t * C + j] = pack(B[k0 : k0 + kn, j], cfg.weight_bits)
        chain = (CHAIN_CONTINUES if t else 0) | (CHAIN_MORE if t < len(starts) - 1 else 0)
        cmds.append({"rs1": a_base + t * R, "rs2": b_base + t * C, "k_dim": kn, "chain": chain})
    return sp, cmds


class TestElaboration:
    def test_rows_must_fit_a_scratchpad_row(self):
        # 4 rows of 8-bit A elements do not fit a 2-column (16-bit) scratchpad row
        with pytest.raises(ValueError, match="cannot hold 4 A elements"):
            ExecuteController(config(grid_rows=4, grid_cols=2))

    def test_needs_a_dataflow(self):
        with pytest.raises(ValueError, match="not implemented"):
            ExecuteController(config(dataflow=Dataflow(0)))

    @pytest.mark.parametrize(
        "dataflow,initial",
        [(Dataflow.OUTPUT_STATIONARY, 0), (Dataflow.B_STATIONARY, 1), (Dataflow.A_STATIONARY, 1)],
    )
    def test_single_mode_is_fixed(self, dataflow, initial):
        harness = Harness(config(dataflow=dataflow))
        log = run(harness, {}, [(ExecOpcode.CONFIG, {"rs1": 1 - initial})])
        assert log["dataflow"] == [initial]

    def test_block_length(self):
        assert ExecuteController(config(grid_cols=4)).block == 4

    def test_verilog(self):
        skip_without_yosys()
        from amaranth.back import verilog

        out = verilog.convert(ExecuteController(config()), name="ExecuteController")
        assert "module ExecuteController" in out


class TestCommands:
    def test_config_selects_the_mode(self):
        """WS and IS are both the stationary mode."""
        log = run(
            Harness(config()),
            {},
            [(ExecOpcode.CONFIG, {"rs1": m}) for m in (WS, OS, IS, OS)],
        )
        assert log["dataflow"] == [1, 0, 1, 0]
        assert [c[1] for c in log["completed"]] == [1, 2, 3, 4]

    def test_unknown_opcode_completes_without_side_effects(self):
        log = run(Harness(config()), {}, [(0x7F, {"rs1": 5, "k_dim": 3})])
        assert [c[1] for c in log["completed"]] == [1]
        assert not (log["a_reads"] or log["b_reads"] or log["writes"] or log["results"])

    def test_os_preload_does_nothing(self):
        log = run(Harness(config()), {}, [(ExecOpcode.PRELOAD, {"rs1": 8, "k_dim": 2})])
        assert [c[1] for c in log["completed"]] == [1]
        assert not (log["a_reads"] or log["b_reads"] or log["writes"])

    def test_compute_zero_k_streams_nothing(self):
        log = run(Harness(config()), {}, [(ExecOpcode.COMPUTE, {"rd": 3, "k_dim": 0})])
        assert [c[1] for c in log["completed"]] == [1]
        assert not (log["a_reads"] or log["writes"] or log["results"])


class TestOutputStationary:
    @pytest.mark.parametrize(
        "opcode,accumulate", [(ExecOpcode.COMPUTE, 0), (ExecOpcode.COMPUTE_ACCUMULATE, 1)]
    )
    def test_one_reduction(self, opcode, accumulate):
        harness = Harness(config())
        R, C = harness.rows, harness.cols
        rng = np.random.default_rng(7)
        A = rng.integers(-128, 128, (R, 2))
        B = rng.integers(-128, 128, (2, C))
        sp, cmds = os_tiles(harness, A, B, 10, 40)
        log = run(harness, sp, [(opcode, {**cmds[0], "rd": 100})])

        assert log["a_reads"] == [10 + i for i in range(R)]
        assert log["b_reads"] == [40 + j for j in range(C)]
        # The result buffers write C row-major, row 0 first
        assert [w[0] for w in log["writes"]] == [100 + i for i in range(R)]
        assert all(w[2] == accumulate for w in log["writes"])
        np.testing.assert_array_equal([w[1] for w in log["writes"]], A @ B)
        assert [c[1] for c in log["completed"]] == [1]
        assert [r[1] for r in log["results"]] == [1]

    @pytest.mark.parametrize("tracked", [False, True])
    def test_reduction_chained_across_commands(self, tracked):
        """C stays in the PEs between the K tiles; only the last command writes it."""
        harness = Harness(config(grid_rows=2, grid_cols=2, tile_rows=2, tile_cols=2))
        rng = np.random.default_rng(8)
        A = rng.integers(-128, 128, (harness.rows, 11))
        B = rng.integers(-128, 128, (11, harness.cols))
        sp, cmds = os_tiles(harness, A, B, 0, 64)
        log = run(harness, sp, [(ExecOpcode.COMPUTE, {**c, "rd": 8}) for c in cmds],
                  tracked=tracked)  # fmt: skip
        out = written(log)
        np.testing.assert_array_equal([out[8 + i] for i in range(harness.rows)], A @ B)
        assert len(log["writes"]) == harness.rows
        assert [r[1] for r in log["results"]] == [len(cmds)]

    def test_tiles_back_to_back_without_bubbles(self):
        """Reductions of K >= 2R - 1 (two K tiles each): the streams never stop."""
        harness = Harness(config(grid_rows=4, grid_cols=4))
        R = harness.rows
        rng = np.random.default_rng(9)
        tiles, sp, cmds, expected = 3, {}, [], []
        for n in range(tiles):
            A = rng.integers(-128, 128, (R, 2 * harness.kb))
            B = rng.integers(-128, 128, (2 * harness.kb, harness.cols))
            tile_sp, tile_cmds = os_tiles(harness, A, B, n * 16, 64 + n * 16)
            sp.update(tile_sp)
            cmds += [(ExecOpcode.COMPUTE, {**c, "rd": n * R}) for c in tile_cmds]
            expected.append(A @ B)
        log = run(harness, sp, cmds, tracked=True)

        out = written(log)
        np.testing.assert_array_equal([out[i] for i in range(tiles * R)], np.vstack(expected))
        # Row 0's stream is valid on every clock from its first element to its last
        s = log["streaming"]
        on = [t for t, v in enumerate(s) if v]
        assert len(on) == tiles * 2 * harness.kb
        assert on[-1] - on[0] + 1 == len(on)

    def test_short_reductions_wait_for_the_drain(self):
        """K < 2R - 1: each next reduction ends 2R - 1 clocks after the last, no collision."""
        harness = Harness(config(grid_rows=4, grid_cols=4))
        R = harness.rows
        rng = np.random.default_rng(10)
        sp, cmds, expected = {}, [], []
        for n in range(3):
            A = rng.integers(-128, 128, (R, 1))
            B = rng.integers(-128, 128, (1, harness.cols))
            tile_sp, tile_cmds = os_tiles(harness, A, B, n * 8, 64 + n * 8)
            sp.update(tile_sp)
            cmds += [(ExecOpcode.COMPUTE, {**c, "rd": n * R}) for c in tile_cmds]
            expected.append(A @ B)
        log = run(harness, sp, cmds, tracked=True)

        out = written(log)
        np.testing.assert_array_equal([out[i] for i in range(3 * R)], np.vstack(expected))
        starts = [t for t, v in enumerate(log["streaming"]) if v]
        assert np.diff(starts).tolist() == [2 * R - 1, 2 * R - 1]

    def test_events_carry_each_commands_sync(self):
        """L2 is released when the last block is in L1, before the results arrive."""
        harness = Harness(config())
        rng = np.random.default_rng(11)
        A = rng.integers(-128, 128, (2, 2))
        B = rng.integers(-128, 128, (2, 2))
        sp, cmds = os_tiles(harness, A, B, 0, 8)
        log = run(harness, sp, [(ExecOpcode.COMPUTE, cmds[0])] * 2, tracked=True)

        assert [(c[1], c[2]) for c in log["completed"]] == [(1, TRACKED), (2, TRACKED | 1)]
        assert [(r[1], r[2]) for r in log["results"]] == [(1, TRACKED), (2, TRACKED | 1)]
        assert log["completed"][0][0] < log["results"][0][0]
        assert log["completed"][1][0] < log["results"][0][0]  # the next tile's L2 is free first


class TestStationary:
    def preload_compute(self, harness, W, A, *, w_base=50, a_base=0, rd=30):
        """Lay out W (row-major) and A (column-major, M tiles) in L2."""
        cfg, R, KB = harness.cfg, harness.rows, harness.kb
        sp = {w_base + r: pack(W[r], cfg.weight_bits) for r in range(W.shape[0])}
        cmds = [(ExecOpcode.PRELOAD, {"rs1": w_base, "k_dim": W.shape[0]})]
        for t, m0 in enumerate(range(0, A.shape[0], KB)):
            mn = min(KB, A.shape[0] - m0)
            for k in range(R):
                col = A[m0 : m0 + mn, k] if k < A.shape[1] else [0] * mn
                sp[a_base + t * R + k] = pack(col, cfg.input_bits)
            cmds.append((ExecOpcode.COMPUTE, {"rs1": a_base + t * R, "rd": rd + m0, "k_dim": mn}))
        return sp, cmds

    @pytest.mark.parametrize("tracked", [False, True])
    def test_preload_and_compute(self, tracked):
        harness = Harness(config(grid_rows=2, grid_cols=2, tile_rows=2, tile_cols=2))
        R, C = harness.rows, harness.cols
        rng = np.random.default_rng(12)
        W = rng.integers(-128, 128, (R, C))
        A = rng.integers(-128, 128, (9, R))
        sp, cmds = self.preload_compute(harness, W, A)
        log = run(harness, sp, [(ExecOpcode.CONFIG, {"rs1": WS}), *cmds], tracked=tracked)

        assert log["b_reads"] == [50 + r for r in reversed(range(R))]  # last row first
        out = written(log)
        np.testing.assert_array_equal([out[30 + m] for m in range(9)], A @ W)

    def test_streams_back_to_back_without_bubbles(self):
        harness = Harness(config(grid_rows=4, grid_cols=4))
        R, C = harness.rows, harness.cols
        rng = np.random.default_rng(13)
        W = rng.integers(-128, 128, (R, C))
        A = rng.integers(-128, 128, (3 * harness.kb, R))
        sp, cmds = self.preload_compute(harness, W, A)
        log = run(harness, sp, [(ExecOpcode.CONFIG, {"rs1": WS}), *cmds], tracked=True)

        out = written(log)
        np.testing.assert_array_equal([out[30 + m] for m in range(A.shape[0])], A @ W)
        on = [t for t, v in enumerate(log["streaming"]) if v]
        assert len(on) == A.shape[0] and on[-1] - on[0] + 1 == len(on)

    def test_preload_waits_for_the_previous_stream(self):
        """New weights after a stream: the first stream's results use the old weights."""
        harness = Harness(config(grid_rows=4, grid_cols=4))
        R, C = harness.rows, harness.cols
        rng = np.random.default_rng(14)
        W1, W2 = rng.integers(-128, 128, (2, R, C))
        A1, A2 = rng.integers(-128, 128, (2, harness.kb, R))
        sp1, cmds1 = self.preload_compute(harness, W1, A1, w_base=50, a_base=0, rd=0)
        sp2, cmds2 = self.preload_compute(harness, W2, A2, w_base=60, a_base=20, rd=10)
        log = run(harness, {**sp1, **sp2}, [(ExecOpcode.CONFIG, {"rs1": IS}), *cmds1, *cmds2],
                  tracked=True)  # fmt: skip
        out = written(log)
        np.testing.assert_array_equal([out[m] for m in range(harness.kb)], A1 @ W1)
        np.testing.assert_array_equal([out[10 + m] for m in range(harness.kb)], A2 @ W2)

    def test_preload_zero_fills_and_clamps(self):
        """k_dim < R: the rows below get zero weights; k_dim > R reads R rows."""
        harness = Harness(config(grid_rows=4, grid_cols=4))
        R, C = harness.rows, harness.cols
        rng = np.random.default_rng(15)
        W = rng.integers(-128, 128, (2, C))
        A = rng.integers(-128, 128, (3, R))  # columns 2..3 meet zero weights
        sp, cmds = self.preload_compute(harness, W, A)
        log = run(harness, sp, [(ExecOpcode.CONFIG, {"rs1": WS}), *cmds])
        out = written(log)
        np.testing.assert_array_equal([out[30 + m] for m in range(3)], A[:, :2] @ W)

        log = run(Harness(config(dataflow=Dataflow.B_STATIONARY)), {},
                  [(ExecOpcode.PRELOAD, {"rs1": 0, "k_dim": 9})])  # fmt: skip
        assert sorted(log["b_reads"]) == [0, 1]
