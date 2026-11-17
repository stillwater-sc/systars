"""
Unit tests for Streamer (the Streamer), wired to a real SystolicArray.
Python models serve the scratchpad's two read ports, or the real banked L2 does
(``Harness(cfg, l2=True)``, #24), and the Combiner writes are recorded. Commands are untracked (one at a time, ``cmd_sync`` 0) or tracked (pushed
back to back, ``cmd_sync`` != 0), where the Streamer overlaps them.
"""

import numpy as np
import pytest
from amaranth import Elaboratable, Module

from systars.config import Dataflow, SystolicConfig
from systars.controller.result_path import ResultPath
from systars.controller.streamer import (
    CHAIN_CONTINUES,
    CHAIN_MORE,
    Streamer,
)
from systars.controller.sync import Sync
from systars.core.systolic_array import SystolicArray
from systars.errors import L1Error
from systars.memory.scratchpad import Scratchpad
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
    """The Streamer, the fabric and the result path (the L1 -> L2 hop, wired as in
    SystolicTop). With ``l2``, the real banked L2 serves the reads: ``b_base`` is then
    its B group's first row, where stationary and B tiles go."""

    def __init__(self, cfg, *, l2=False):
        self.cfg = cfg
        self.ctrl = Streamer(cfg)
        self.results = ResultPath(cfg)
        self.array = SystolicArray(cfg)
        self.rows, self.cols, self.kb = self.ctrl.rows, self.ctrl.cols, self.ctrl.block
        self.l2 = Scratchpad(cfg) if l2 else None
        self.b_base = cfg.l2_group_rows if l2 else 1000

    def elaborate(self, _platform):
        m = Module()
        m.submodules.ctrl = ctrl = self.ctrl
        m.submodules.results = res = self.results
        m.submodules.array = array = self.array
        m.d.comb += [
            res.tile_valid.eq(ctrl.tile_valid),
            ctrl.tile_ready.eq(res.tile_ready),
            *[getattr(res, f"tile_{name}").eq(getattr(ctrl, f"tile_{name}"))
              for name, _ in res.layout],
            res.stationary.eq(ctrl.cfg_dataflow),
            ctrl.results_busy.eq(res.busy),
            ctrl.held.eq(res.held),
            ctrl.held_id.eq(res.held_id),
            ctrl.held_sync.eq(res.held_sync),
        ]  # fmt: skip
        if self.l2 is not None:
            m.submodules.l2 = l2 = self.l2
            for port, prefix in (("a", "read"), ("b", "read_b")):
                m.d.comb += [
                    getattr(l2, f"{prefix}_req").eq(getattr(ctrl, f"sp_{port}_read_req")),
                    getattr(l2, f"{prefix}_addr").eq(getattr(ctrl, f"sp_{port}_read_addr")),
                    getattr(ctrl, f"sp_{port}_read_data").eq(getattr(l2, f"{prefix}_data")),
                    getattr(ctrl, f"sp_{port}_read_valid").eq(getattr(l2, f"{prefix}_valid")),
                ]
        m.d.comb += [
            array.in_control_dataflow.eq(ctrl.array_in_dataflow),
            res.array_out_collision.eq(array.out_collision),
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
                getattr(array, f"in_w_valid_{j}").eq(getattr(ctrl, f"array_in_w_valid_{j}")),
            ]
            for name in ("c", "c_valid", "psum", "psum_valid"):
                m.d.comb += getattr(res, f"array_out_{name}_{j}").eq(
                    getattr(array, f"out_{name}_{j}")
                )
        return m


def pack(values, bits):
    return sum((int(v) & ((1 << bits) - 1)) << (i * bits) for i, v in enumerate(values))


def unpack(word, bits, n):
    vals = [(word >> (i * bits)) & ((1 << bits) - 1) for i in range(n)]
    return [v - (1 << bits) if v >> (bits - 1) else v for v in vals]


def run(harness, scratchpad, commands, *, tracked=False, max_cycles=4000, expect_error=False):
    """Issue ``commands`` (opcode, fields) and log reads, writes, events and streams.

    Untracked: one command at a time, each waited for. Tracked: pushed back to back
    (``cmd_ready`` only), then the run waits for every completion and result. A
    command's ``sync`` field overrides its ``cmd_sync``.
    """
    ctrl, rp, cfg = harness.ctrl, harness.results, harness.cfg
    log = {
        "a_reads": [], "b_reads": [], "writes": [], "completed": [], "results": [],
        "dataflow": [], "streaming": [], "accepted": [], "error": 0,
        "a_clocks": [], "b_clocks": [],
    }  # fmt: skip

    def port_model(port):
        async def process(ctx):
            pipe = [None] * SP_LATENCY
            t = 0
            async for _clk, _, req, addr in ctx.tick().sample(
                getattr(ctrl, f"sp_{port}_read_req"), getattr(ctrl, f"sp_{port}_read_addr")
            ):
                t += 1
                if req:
                    log[f"{port}_reads"].append(addr)
                    log[f"{port}_clocks"].append(t)
                if harness.l2 is not None:
                    continue  # the real L2 answers
                pipe = [addr if req else None, *pipe[:-1]]
                out = pipe[-1]
                ctx.set(getattr(ctrl, f"sp_{port}_read_valid"), out is not None)
                ctx.set(getattr(ctrl, f"sp_{port}_read_data"), scratchpad.get(out, 0))

        return process

    async def monitor(ctx):
        """Log every clock's Combiner writes, completions, results and accepts."""
        t = 0
        async for sample in ctx.tick().sample(
            rp.combiner_write_req, rp.combiner_write_addr, rp.combiner_write_data, rp.combiner_accumulate,
            ctrl.completed, ctrl.completed_id, ctrl.completed_sync,
            rp.result, rp.result_id, rp.result_sync,
            ctrl.array_in_valid_0, ctrl.error | rp.error, ctrl.cmd_valid, ctrl.cmd_ready,
            ctrl.cmd_id,
        ):  # fmt: skip
            _, _, req, addr, data, acc, done, done_id, done_sync, res, res_id, res_sync = sample[
                :12
            ]
            streaming, error, valid, ready, cmd_id = sample[12:17]
            if valid and ready:
                log["accepted"].append((t, cmd_id))
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
        if harness.l2 is not None:  # the rows, straight into the banks
            lay, n_banks = harness.l2.layout, harness.l2.layout.banks
            for row, value in scratchpad.items():
                group, bank, bank_row = lay.place(row)
                ctx.set(harness.l2.banks[group * n_banks + bank].mem.data[bank_row], value)
        for n, (opcode, fields) in enumerate(commands):
            ctx.set(ctrl.cmd_opcode, opcode)
            ctx.set(ctrl.cmd_id, n + 1)
            for name in ("rs1", "rs2", "rd", "k_dim", "chain"):
                ctx.set(getattr(ctrl, f"cmd_{name}"), fields.get(name, 0))
            ctx.set(ctrl.cmd_sync, fields.get("sync", TRACKED | n if tracked else 0))
            ctx.set(ctrl.cmd_valid, 1)
            await ctx.tick().until(ctrl.cmd_ready)
            ctx.set(ctrl.cmd_valid, 0)
            if not tracked:
                await ctx.tick().until(ctrl.completed)
                log["dataflow"].append(ctx.get(ctrl.cfg_dataflow))
        while ctx.get(ctrl.busy) or len(log["completed"]) < len(commands):
            await ctx.tick()
        await ctx.tick().repeat(2)
        # The Streamer's error, else the result path's (SystolicTop's order)
        log["error_kind"] = ctx.get(ctrl.error_kind if ctx.get(ctrl.error) else rp.error_kind)
        if harness.l2 is not None:
            log["l2_error"] = ctx.get(harness.l2.error)

    run_sim(
        harness,
        testbench,
        processes=[port_model("a"), port_model("b"), monitor],
        max_cycles=max_cycles,
    )
    assert log["error"] == int(expect_error)
    return log


def written(log):
    """The Combiner, as written: address -> row (the last write wins)."""
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
            Streamer(config(grid_rows=4, grid_cols=2))

    def test_needs_a_dataflow(self):
        with pytest.raises(ValueError, match="not implemented"):
            Streamer(config(dataflow=Dataflow(0)))

    @pytest.mark.parametrize(
        "dataflow,initial",
        [(Dataflow.OUTPUT_STATIONARY, 0), (Dataflow.B_STATIONARY, 1), (Dataflow.A_STATIONARY, 1)],
    )
    def test_single_mode_is_fixed(self, dataflow, initial):
        harness = Harness(config(dataflow=dataflow))
        log = run(harness, {}, [(ExecOpcode.CONFIG, {"rs1": 1 - initial})])
        assert log["dataflow"] == [initial]

    def test_block_length(self):
        assert Streamer(config(grid_cols=4)).block == 4

    def test_verilog(self):
        skip_without_yosys()
        from amaranth.back import verilog

        out = verilog.convert(Streamer(config()), name="Streamer")
        assert "module Streamer" in out


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


class TestSignalNeedsAResult:
    """The Combiner token is emitted on the result event: a command with SIGNAL
    that writes no result would leave its slot OWNED, so the Streamer flags it."""

    SIGNAL = TRACKED | Sync.encode(dst=1, acquire=True, signal=True)

    @pytest.mark.parametrize(
        "opcode,fields",
        [
            (ExecOpcode.COMPUTE, {"rd": 3, "k_dim": 0}),
            (ExecOpcode.COMPUTE, {"k_dim": 2, "chain": CHAIN_MORE}),
            (ExecOpcode.PRELOAD, {"rs1": 8, "k_dim": 2}),
            (ExecOpcode.CONFIG, {"rs1": OS}),
            (0x7F, {}),
        ],
        ids=["zero_k", "chain_more", "os_preload", "config", "unknown"],
    )
    def test_signal_without_a_result_is_an_error(self, opcode, fields):
        log = run(Harness(config()), {}, [(opcode, {**fields, "sync": self.SIGNAL})],
                  tracked=True, expect_error=True)  # fmt: skip
        assert not log["results"]
        assert log["error_kind"] == L1Error.SIGNAL_NO_RESULT

    def test_signal_on_the_result_is_not(self):
        harness = Harness(config())
        rng = np.random.default_rng(3)
        A = rng.integers(-128, 128, (harness.rows, 2))
        B = rng.integers(-128, 128, (2, harness.cols))
        sp, cmds = os_tiles(harness, A, B, 10, 40)
        log = run(harness, sp, [(ExecOpcode.COMPUTE, {**cmds[0], "rd": 0, "sync": self.SIGNAL})],
                  tracked=True)  # fmt: skip
        assert [(r[1], r[2]) for r in log["results"]] == [(1, self.SIGNAL)]


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
        """Lay out W and A (both column-major; A in M tiles) in L2."""
        cfg, R, KB = harness.cfg, harness.rows, harness.kb
        sp = {w_base + j: pack(W[:, j], cfg.weight_bits) for j in range(W.shape[1])}
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
        """A column-major PRELOAD then a COMPUTE gives ``A @ W`` in WS."""
        harness = Harness(config(grid_rows=2, grid_cols=2, tile_rows=2, tile_cols=2))
        R, C = harness.rows, harness.cols
        rng = np.random.default_rng(12)
        W = rng.integers(-128, 128, (R, C))
        A = rng.integers(-128, 128, (9, R))
        sp, cmds = self.preload_compute(harness, W, A)
        log = run(harness, sp, [(ExecOpcode.CONFIG, {"rs1": WS}), *cmds], tracked=tracked)

        assert log["b_reads"] == [50 + j for j in range(C)]  # one column per read
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

    def test_new_weights_load_behind_the_stream(self):
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

    def weight_tiles(self, harness, n, m, *, seed, mode=WS, reuse=False):
        """``n`` tiles, each PRELOAD W_t then COMPUTE A_t (``m`` rows), tracked. With
        ``reuse`` every tile adds a second COMPUTE under the same weights."""
        R, C = harness.rows, harness.cols
        rng = np.random.default_rng(seed)
        Ws = rng.integers(-128, 128, (n, R, C))
        As = rng.integers(-128, 128, (n, 2 if reuse else 1, m, R))
        sp, cmds, expect = {}, [(ExecOpcode.CONFIG, {"rs1": mode})], {}
        for t in range(n):
            w_base, a_base = harness.b_base + t * C, t * 2 * R
            sp.update({w_base + j: pack(Ws[t][:, j], harness.cfg.weight_bits) for j in range(C)})
            cmds.append((ExecOpcode.PRELOAD, {"rs1": w_base, "k_dim": R}))
            for u, A in enumerate(As[t]):
                base, rd = a_base + u * R, 100 * t + 50 * u
                sp.update({base + k: pack(A[:, k], harness.cfg.input_bits) for k in range(R)})
                cmds.append((ExecOpcode.COMPUTE, {"rs1": base, "rd": rd, "k_dim": m}))
                expect[rd] = A @ Ws[t]
        log = run(harness, sp, cmds, tracked=True)
        out = written(log)
        for rd, rows in expect.items():
            np.testing.assert_array_equal([out[rd + r] for r in range(m)], rows)
        return log

    @pytest.mark.parametrize("mode", [WS, IS])
    @pytest.mark.parametrize("shape", [(1, 1, 2, 2), (2, 2, 1, 1), (1, 1, 2, 3), (2, 2, 2, 2)])
    def test_new_weight_tiles_stream_back_to_back(self, shape, mode):
        """PRELOAD, COMPUTE, PRELOAD, COMPUTE, ...: with M >= max(R, C) the weights
        of the next tile load behind the stream, and the streams have no bubbles."""
        tr, tc, gr, gc = shape
        harness = Harness(config(tile_rows=tr, tile_cols=tc, grid_rows=gr, grid_cols=gc))
        m = max(harness.rows, harness.cols)
        log = self.weight_tiles(harness, 4, m, seed=16, mode=mode)
        on = [t for t, v in enumerate(log["streaming"]) if v]
        assert len(on) == 4 * m and on[-1] - on[0] + 1 == len(on)

    def test_short_streams_wait_for_the_weights(self):
        """M < R: a stream cannot outrun the weights it swaps in; results stay right."""
        harness = Harness(config(grid_rows=4, grid_cols=4))
        self.weight_tiles(harness, 3, 1, seed=17)
        self.weight_tiles(harness, 3, 2, seed=18)

    def test_reused_weights_are_not_swapped_again(self):
        """A second COMPUTE under the same weights keeps them while the next load."""
        harness = Harness(config(grid_rows=4, grid_cols=4))
        log = self.weight_tiles(harness, 3, harness.rows, seed=19, reuse=True)
        on = [t for t, v in enumerate(log["streaming"]) if v]
        assert on[-1] - on[0] + 1 == len(on)


class TestOnTheBankedL2:
    """#24: the Streamer's schedules on the real banked L2, A and stationary tiles in
    the A group, B and weight tiles in the B group. Two reads a clock, no stall, no
    bank conflict."""

    def harness(self, **fields):
        return Harness(config(**{"sp_capacity_kb": 1, "l2_macro_rows": 32, **fields}), l2=True)

    def test_os_tiles_back_to_back(self):
        harness = self.harness(grid_rows=4, grid_cols=4)
        R = harness.rows
        rng = np.random.default_rng(24)
        sp, cmds, expected = {}, [], []
        for n in range(3):
            A = rng.integers(-128, 128, (R, 2 * harness.kb))
            B = rng.integers(-128, 128, (2 * harness.kb, harness.cols))
            tile_sp, tile_cmds = os_tiles(harness, A, B, n * 16, harness.b_base + n * 16)
            sp.update(tile_sp)
            cmds += [(ExecOpcode.COMPUTE, {**c, "rd": n * R}) for c in tile_cmds]
            expected.append(A @ B)
        log = run(harness, sp, cmds, tracked=True)
        out = written(log)
        np.testing.assert_array_equal([out[i] for i in range(3 * R)], np.vstack(expected))
        on = [t for t, v in enumerate(log["streaming"]) if v]
        assert len(on) == 3 * 2 * harness.kb and on[-1] - on[0] + 1 == len(on)
        assert set(log["a_clocks"]) == set(log["b_clocks"])  # an A and a B row every clock
        assert log["l2_error"] == 0

    @pytest.mark.parametrize("mode", [WS, IS])
    @pytest.mark.parametrize("occupancy", [1, 2, 3])
    def test_new_weight_tiles_stream_back_to_back(self, mode, occupancy):
        """PRELOAD reads the B group while COMPUTE streams the A group, every tile,
        whatever the banks' occupancy: the streams have no bubbles."""
        # l2_macro_rows=0: the banks come from the occupancy alone (3 -> 4 under a
        # 4x4 array, so the wrap from one tile to the next never meets a busy bank)
        harness = self.harness(
            grid_rows=4, grid_cols=4, l2_bank_occupancy=occupancy, l2_macro_rows=0
        )
        m = max(harness.rows, harness.cols)
        log = TestStationary().weight_tiles(harness, 4, m, seed=16, mode=mode)
        on = [t for t, v in enumerate(log["streaming"]) if v]
        assert len(on) == 4 * m and on[-1] - on[0] + 1 == len(on)
        # The two ports read in the same clocks: one per group, never both in one
        g = harness.cfg.l2_group_rows
        assert all(a < g for a in log["a_reads"]) and all(b >= g for b in log["b_reads"])
        assert set(log["a_clocks"]) & set(log["b_clocks"]), "PRELOAD never ran beside a stream"
        assert log["l2_error"] == 0


class TestWavefrontSchedule:
    """``systars.wavefront.streamer_starts`` (the schedule the wavefront animations
    are recorded with) is the schedule the Streamer runs."""

    @staticmethod
    def stream_starts(log, lengths):
        """The clock each tile's stream starts on lane 0 (its K or M clocks are valid)."""
        on = [t for t, v in enumerate(log["streaming"]) if v]
        assert len(on) == sum(lengths)
        return [on[sum(lengths[:n])] for n in range(len(lengths))]

    @pytest.mark.parametrize("lengths", [[1, 1, 1], [8, 8, 8], [8, 3, 4], [4, 7, 2]])
    def test_output_stationary(self, lengths):
        from systars.wavefront import streamer_starts

        harness = Harness(config(grid_rows=4, grid_cols=4))
        R, C = harness.rows, harness.cols
        rng = np.random.default_rng(sum(lengths))
        sp, cmds = {}, []
        for n, k in enumerate(lengths):
            A = rng.integers(-128, 128, (R, k))
            B = rng.integers(-128, 128, (k, C))
            tile_sp, tile_cmds = os_tiles(harness, A, B, n * 16, 64 + n * 16)
            sp.update(tile_sp)
            cmds += [(ExecOpcode.COMPUTE, {**c, "rd": n * R}) for c in tile_cmds]
        log = run(harness, sp, cmds, tracked=True)

        starts = self.stream_starts(log, lengths)
        model = streamer_starts(0, R, C, lengths)
        assert np.diff(starts).tolist() == np.diff(model).tolist()

    @pytest.mark.parametrize("lengths", [[4, 4, 4], [4, 2, 4], [1, 3, 2]])
    def test_stationary(self, lengths):
        from systars.wavefront import streamer_starts

        harness = Harness(config(grid_rows=4, grid_cols=4))
        R, C = harness.rows, harness.cols
        rng = np.random.default_rng(sum(lengths) + 1)
        W = rng.integers(-128, 128, (R, C))
        sp = {50 + r: pack(W[r], harness.cfg.weight_bits) for r in range(R)}
        cmds = [(ExecOpcode.CONFIG, {"rs1": WS}), (ExecOpcode.PRELOAD, {"rs1": 50, "k_dim": R})]
        for n, m_len in enumerate(lengths):
            A = rng.integers(-128, 128, (m_len, R))
            for k in range(R):
                sp[n * R + k] = pack(A[:, k], harness.cfg.input_bits)
            cmds.append((ExecOpcode.COMPUTE, {"rs1": n * R, "rd": 8 * n, "k_dim": m_len}))
        log = run(harness, sp, cmds, tracked=True)

        starts = self.stream_starts(log, lengths)
        model = streamer_starts(1, R, C, lengths)
        assert np.diff(starts).tolist() == np.diff(model).tolist()
