"""
Processing Element (PE) - the multiply-accumulate unit of the systolic array.

A PE supports two dataflows, selected at run time by ``in_control_dataflow``:

- **Output-stationary (OS, 0).** A flows right and B flows down, both passing
  through combinationally. While ``in_valid``, the PE accumulates in place:
  ``c += a * b``. The results leave by shifting down the column.
- **Weight-stationary (WS, 1).** The PE holds a weight ``w`` (an element of B).
  A flows right, and a partial sum flows down: ``psum_out = psum_in + a * w``.

Both dataflows share one **shift chain** per column, ``in_d`` -> ``out_c``. While
``in_shift`` is high, the PE's chain register loads ``in_d`` (the register of
the PE above):

- OS: the chain register is ``c``. Shifting drains the results down the column,
  bottom row first, and shifts in ``in_d`` (zero, from the top) behind them.
- WS: the chain register is ``w``. Shifting loads the weights, last row first.

The PE has no pass-through registers: ``out_a``, ``out_b`` and ``out_psum`` are
combinational, and ``out_c`` is the chain register. Pipeline registers sit only
at PEArray boundaries, in ``SystolicArray``.
"""

from amaranth import Module, Mux, Signal, signed
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig

OUTPUT_STATIONARY = 0
WEIGHT_STATIONARY = 1


class PE(Component):
    """
    Processing Element.

    Ports:
        in_a, out_a: A operand (flows right, combinational)
        in_b, out_b: B operand (OS; flows down, combinational)
        in_psum, out_psum: Partial sum (WS; flows down, ``in_psum + a * w``)
        in_d, out_c: Shift chain (``out_c`` is ``c`` in OS, ``w`` in WS)
        in_valid: Accumulate ``a * b`` into ``c`` this cycle (OS)
        in_shift: Load the chain register from ``in_d`` this cycle
        in_control_dataflow: 0 = output-stationary, 1 = weight-stationary
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        super().__init__(
            {
                "in_a": In(signed(config.input_bits)),
                "in_b": In(signed(config.weight_bits)),
                "in_psum": In(signed(config.acc_bits)),
                "in_d": In(signed(config.acc_bits)),
                "in_valid": In(1),
                "in_shift": In(1),
                "in_control_dataflow": In(1),
                "out_a": Out(signed(config.input_bits)),
                "out_b": Out(signed(config.weight_bits)),
                "out_psum": Out(signed(config.acc_bits)),
                "out_c": Out(signed(config.acc_bits)),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        c = Signal(signed(cfg.acc_bits), name="c")  # OS accumulator
        w = Signal(signed(cfg.weight_bits), name="w")  # WS stationary weight
        ws = self.in_control_dataflow

        product = Signal(signed(cfg.input_bits + cfg.weight_bits), name="product")
        m.d.comb += product.eq(self.in_a * Mux(ws, w, self.in_b))

        m.d.comb += [
            self.out_a.eq(self.in_a),
            self.out_b.eq(self.in_b),
            self.out_psum.eq(self.in_psum + product),
            self.out_c.eq(Mux(ws, w, c)),
        ]

        with m.If(self.in_shift):
            with m.If(ws):
                m.d.sync += w.eq(self.in_d[: cfg.weight_bits].as_signed())
            with m.Else():
                m.d.sync += c.eq(self.in_d)
        with m.Elif(self.in_valid & ~ws):
            m.d.sync += c.eq(c + product)

        return m
