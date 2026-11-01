"""
Unit tests for ExecuteController, wired to a real SystolicArray. Python models
serve the scratchpad's two read ports and record the accumulator writes.
"""

import numpy as np
import pytest
from amaranth import Elaboratable, Module

from systars.config import Dataflow, SystolicConfig
from systars.controller.execute import ExecuteController
from systars.core.systolic_array import SystolicArray
from systars.util.commands import ExecOpcode

from ._sim import run_sim, skip_without_yosys

SP_LATENCY = 3


def config(**overrides):
    fields = {"grid_rows": 2, "grid_cols": 2, "tile_rows": 1, "tile_cols": 1}
    fields.update(overrides)
    return SystolicConfig(**fields)


class Harness(Elaboratable):
    def __init__(self, cfg):
        self.cfg = cfg
        self.rows = cfg.grid_rows * cfg.tile_rows
        self.cols = cfg.grid_cols * cfg.tile_cols
        self.ctrl = ExecuteController(cfg)
        self.array = SystolicArray(cfg)

    def elaborate(self, _platform):
        m = Module()
        m.submodules.ctrl = ctrl = self.ctrl
        m.submodules.array = array = self.array
        m.d.comb += [
            array.in_valid.eq(ctrl.array_in_valid),
            array.in_last.eq(ctrl.array_in_last),
            array.in_shift.eq(ctrl.array_in_shift),
            array.in_control_dataflow.eq(ctrl.array_in_dataflow),
            ctrl.array_out_valid.eq(array.out_valid),
            ctrl.array_out_last.eq(array.out_last),
        ]
        for i in range(self.rows):
            m.d.comb += getattr(array, f"in_a_{i}").eq(getattr(ctrl, f"array_in_a_{i}"))
        for j in range(self.cols):
            m.d.comb += [
                getattr(array, f"in_b_{j}").eq(getattr(ctrl, f"array_in_b_{j}")),
                getattr(array, f"in_d_{j}").eq(getattr(ctrl, f"array_in_d_{j}")),
                getattr(ctrl, f"array_out_psum_{j}").eq(getattr(array, f"out_psum_{j}")),
                getattr(ctrl, f"array_out_c_{j}").eq(getattr(array, f"out_c_{j}")),
            ]
        return m


def pack(values, bits):
    return sum((int(v) & ((1 << bits) - 1)) << (i * bits) for i, v in enumerate(values))


def unpack(word, bits, n):
    vals = [(word >> (i * bits)) & ((1 << bits) - 1) for i in range(n)]
    return [v - (1 << bits) if v >> (bits - 1) else v for v in vals]


def run(harness, scratchpad, commands):
    """Issue ``commands`` (opcode, fields) and collect reads, writes and completions."""
    ctrl, cfg = harness.ctrl, harness.cfg
    log = {"a_reads": [], "b_reads": [], "writes": [], "completed": [], "dataflow": []}

    def port_model(port):
        async def process(ctx):
            pipe = [None] * SP_LATENCY
            async for _, _, req, addr in ctx.tick().sample(
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
        async for _, _, req, addr, data, acc, done, done_id in ctx.tick().sample(
            ctrl.acc_write_req,
            ctrl.acc_write_addr,
            ctrl.acc_write_data,
            ctrl.acc_accumulate,
            ctrl.completed,
            ctrl.completed_id,
        ):
            if req:
                log["writes"].append((addr, unpack(data, cfg.acc_bits, harness.cols), acc))
            if done:
                log["completed"].append(done_id)

    async def testbench(ctx):
        for n, (opcode, fields) in enumerate(commands):
            ctx.set(ctrl.cmd_opcode, opcode)
            ctx.set(ctrl.cmd_id, n + 1)
            for name in ("rs1", "rs2", "rd", "k_dim"):
                ctx.set(getattr(ctrl, f"cmd_{name}"), fields.get(name, 0))
            ctx.set(ctrl.cmd_valid, 1)
            await ctx.tick().until(ctrl.cmd_ready)
            ctx.set(ctrl.cmd_valid, 0)
            await ctx.tick().until(ctrl.completed)
            log["dataflow"].append(ctx.get(ctrl.cfg_dataflow))
        await ctx.tick().repeat(2)

    run_sim(
        harness,
        testbench,
        processes=[port_model("a"), port_model("b"), monitor],
        max_cycles=2000,
    )
    return log


class TestElaboration:
    def test_rows_must_fit_a_scratchpad_row(self):
        # 4 rows of 8-bit A elements do not fit a 2-column (16-bit) scratchpad row
        with pytest.raises(ValueError, match="cannot hold 4 A elements"):
            ExecuteController(config(grid_rows=4, grid_cols=2))

    def test_needs_os_or_ws(self):
        with pytest.raises(ValueError, match="not implemented"):
            ExecuteController(config(dataflow=Dataflow.A_STATIONARY))

    @pytest.mark.parametrize(
        "dataflow,initial", [(Dataflow.OUTPUT_STATIONARY, 0), (Dataflow.B_STATIONARY, 1)]
    )
    def test_single_dataflow_is_fixed(self, dataflow, initial):
        harness = Harness(config(dataflow=dataflow))
        log = run(harness, {}, [(ExecOpcode.CONFIG, {"rs1": 1 - initial})])
        assert log["dataflow"] == [initial]

    def test_verilog(self):
        skip_without_yosys()
        from amaranth.back import verilog

        out = verilog.convert(ExecuteController(config()), name="ExecuteController")
        assert "module ExecuteController" in out


class TestCommands:
    def test_config_selects_dataflow(self):
        log = run(
            Harness(config()),
            {},
            [(ExecOpcode.CONFIG, {"rs1": 1}), (ExecOpcode.CONFIG, {"rs1": 0})],
        )
        assert log["dataflow"] == [1, 0]
        assert log["completed"] == [1, 2]

    def test_unknown_opcode_completes_without_side_effects(self):
        log = run(Harness(config()), {}, [(0x7F, {"rs1": 5, "k_dim": 3})])
        assert log["completed"] == [1]
        assert not (log["a_reads"] or log["b_reads"] or log["writes"])

    def test_os_preload_does_nothing(self):
        log = run(Harness(config()), {}, [(ExecOpcode.PRELOAD, {"rs1": 8, "k_dim": 2})])
        assert log["completed"] == [1]
        assert not (log["a_reads"] or log["b_reads"] or log["writes"])


class TestOutputStationary:
    @pytest.mark.parametrize(
        "opcode,accumulate", [(ExecOpcode.COMPUTE, 0), (ExecOpcode.COMPUTE_ACCUMULATE, 1)]
    )
    def test_compute_runs_to_completion(self, opcode, accumulate):
        harness = Harness(config())
        cfg, rows, cols = harness.cfg, harness.rows, harness.cols
        rng = np.random.default_rng(7)
        k = 4
        A = rng.integers(-128, 128, (rows, k))
        B = rng.integers(-128, 128, (k, cols))
        scratchpad = {10 + n: pack(A[:, n], cfg.input_bits) for n in range(k)}
        scratchpad.update({40 + n: pack(B[n], cfg.weight_bits) for n in range(k)})

        log = run(harness, scratchpad, [(opcode, {"rs1": 10, "rs2": 40, "rd": 100, "k_dim": k})])

        assert log["a_reads"] == list(range(10, 10 + k))
        assert log["b_reads"] == list(range(40, 40 + k))
        # Drained bottom row first, one accumulator write per row
        assert [w[0] for w in log["writes"]] == [100 + r for r in reversed(range(rows))]
        assert all(w[2] == accumulate for w in log["writes"])
        C = np.array([w[1] for w in sorted(log["writes"])])
        np.testing.assert_array_equal(C, A @ B)
        assert log["completed"] == [1]

    def test_zero_k_writes_zero_rows(self):
        harness = Harness(config())
        log = run(harness, {}, [(ExecOpcode.COMPUTE, {"rd": 3, "k_dim": 0})])
        assert sorted(w[0] for w in log["writes"]) == [3, 4]
        assert all(w[1] == [0, 0] for w in log["writes"])
        assert log["completed"] == [1]

    def test_back_to_back_computes_start_clear(self):
        """Draining clears the array, so a second COMPUTE does not see the first."""
        harness = Harness(config())
        cfg = harness.cfg
        A1, B1 = np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])
        A2, B2 = np.array([[-1, 0], [2, 1]]), np.array([[3, -3], [1, 1]])
        sp = {}
        for base, A, B in ((0, A1, B1), (20, A2, B2)):
            sp.update({base + n: pack(A[:, n], cfg.input_bits) for n in range(2)})
            sp.update({base + 10 + n: pack(B[n], cfg.weight_bits) for n in range(2)})
        log = run(
            harness,
            sp,
            [
                (ExecOpcode.COMPUTE, {"rs1": 0, "rs2": 10, "rd": 0, "k_dim": 2}),
                (ExecOpcode.COMPUTE, {"rs1": 20, "rs2": 30, "rd": 8, "k_dim": 2}),
            ],
        )
        rows = {w[0]: w[1] for w in log["writes"]}
        np.testing.assert_array_equal([rows[0], rows[1]], A1 @ B1)
        np.testing.assert_array_equal([rows[8], rows[9]], A2 @ B2)


class TestWeightStationary:
    def test_preload_and_compute(self):
        harness = Harness(config(grid_rows=2, grid_cols=2, tile_rows=2, tile_cols=2))
        cfg, rows, cols = harness.cfg, harness.rows, harness.cols
        rng = np.random.default_rng(11)
        m_rows = 5
        W = rng.integers(-128, 128, (rows, cols))
        A = rng.integers(-128, 128, (m_rows, rows))
        sp = {50 + r: pack(W[r], cfg.weight_bits) for r in range(rows)}
        sp.update({n: pack(A[n], cfg.input_bits) for n in range(m_rows)})

        log = run(
            harness,
            sp,
            [
                (ExecOpcode.CONFIG, {"rs1": 1}),
                (ExecOpcode.PRELOAD, {"rs1": 50, "k_dim": rows}),
                (ExecOpcode.COMPUTE, {"rs1": 0, "rd": 30, "k_dim": m_rows}),
            ],
        )

        assert log["b_reads"] == [50 + r for r in reversed(range(rows))]  # last row first
        assert log["a_reads"] == list(range(m_rows))
        assert [w[0] for w in log["writes"]] == list(range(30, 30 + m_rows))
        np.testing.assert_array_equal(np.array([w[1] for w in log["writes"]]), A @ W)
        assert log["completed"] == [1, 2, 3]

    def test_preload_clamps_to_pe_rows(self):
        harness = Harness(config())
        log = run(
            harness,
            {},
            [(ExecOpcode.CONFIG, {"rs1": 1}), (ExecOpcode.PRELOAD, {"rs1": 0, "k_dim": 9})],
        )
        assert log["b_reads"] == [1, 0]

    def test_compute_zero_rows(self):
        harness = Harness(config())
        log = run(
            harness, {}, [(ExecOpcode.CONFIG, {"rs1": 1}), (ExecOpcode.COMPUTE, {"k_dim": 0})]
        )
        assert log["completed"] == [1, 2]
        assert not log["writes"]
