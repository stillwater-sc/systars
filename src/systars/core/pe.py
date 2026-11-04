"""
Processing Element (PE) - the multiply-accumulate unit of the systolic array.

Every PE has an input register on each of its links, so the systolic clock
separates every PE from its neighbours: a value moves one PE per clock.

- **West -> east:** the A operand, with three markers that travel with it:
  ``valid``, ``first`` (the first product of a reduction) and ``last`` (its last).
- **North -> south:** the B operand (OS) and the partial sum (stationary mode).
- **South -> north:** C results (OS), with a ``valid`` bit.

The registers are the PE's ``a``, ``b``, ``psum`` and ``c_up`` (with their markers);
its outputs to the east, south and north neighbours are driven from them, and the
neighbour's input register latches them. A PE computes on the values in its
registers.

Two modes, selected at run time by ``in_control_dataflow``:

- **Output-stationary (OS, 0).** The PE accumulates ``c += a * b`` while ``a`` is
  valid. ``first`` overwrites instead (``c = a * b``), so a reduction needs no
  clear. On ``last`` the PE launches ``c + a * b`` north: it drives ``out_c`` with
  it, and the PE to the north latches it into its ``c_up`` register. Otherwise
  ``out_c`` carries the ``c_up`` register, so results move north one PE per clock
  to the north edge. A PE launching while ``c_up`` holds a result is a collision
  (``out_collision``); the Streamer's drain rule prevents it.
- **Stationary (1).** The PE holds one stationary operand ``w`` (WS: an element of
  B; IS: an element of Aᵀ). The streamed operand flows east in ``a``, and the
  partial sum flows south: ``out_psum = psum + a * w`` while ``a`` is valid.

The stationary register is loaded through a chain down each column: while
``in_shift`` is high, ``w`` loads ``in_w`` (the ``w`` of the PE to the north, or
the column's input at the north edge). The chain is registered PE to PE; only the
shift enable is broadcast.
"""

from amaranth import Module, Mux, Signal, signed
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig

OUTPUT_STATIONARY = 0
STATIONARY = 1
WEIGHT_STATIONARY = STATIONARY  # WS and IS are the same PE mode


class PE(Component):
    """
    Processing Element with an input register on every link.

    Ports:
        in_a, in_valid, in_first, in_last: A operand and its markers (from the west)
        out_a, out_valid, out_first, out_last: The registered A and markers (to the east)
        in_b, out_b: B operand (OS; from the north, registered, to the south)
        in_psum, out_psum: Partial sum (stationary; ``out_psum = psum + a * w``)
        out_psum_valid: ``out_psum`` includes a valid product (the A marker)
        in_c, in_c_valid: C result from the PE to the south (OS)
        out_c, out_c_valid: C result to the PE to the north (OS)
        in_w, out_w: Stationary-register chain (``out_w`` is ``w``)
        in_shift: Load ``w`` from ``in_w`` this cycle
        in_control_dataflow: 0 = output-stationary, 1 = stationary
        out_collision: A launch met a result passing north (OS; never in a valid schedule)
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        acc = signed(config.acc_bits)
        super().__init__(
            {
                "in_a": In(signed(config.input_bits)),
                "in_valid": In(1),
                "in_first": In(1),
                "in_last": In(1),
                "in_b": In(signed(config.weight_bits)),
                "in_psum": In(acc),
                "in_c": In(acc),
                "in_c_valid": In(1),
                "in_w": In(signed(config.weight_bits)),
                "in_shift": In(1),
                "in_control_dataflow": In(1),
                "out_a": Out(signed(config.input_bits)),
                "out_valid": Out(1),
                "out_first": Out(1),
                "out_last": Out(1),
                "out_b": Out(signed(config.weight_bits)),
                "out_psum": Out(acc),
                "out_psum_valid": Out(1),
                "out_c": Out(acc),
                "out_c_valid": Out(1),
                "out_w": Out(signed(config.weight_bits)),
                "out_collision": Out(1),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config
        acc = signed(cfg.acc_bits)

        # Input registers, one per link
        a = Signal(signed(cfg.input_bits), name="a")
        valid = Signal(name="valid")
        first = Signal(name="first")
        last = Signal(name="last")
        b = Signal(signed(cfg.weight_bits), name="b")
        psum = Signal(acc, name="psum")
        c_up = Signal(acc, name="c_up")
        c_up_valid = Signal(name="c_up_valid")
        m.d.sync += [
            a.eq(self.in_a),
            valid.eq(self.in_valid),
            first.eq(self.in_first),
            last.eq(self.in_last),
            b.eq(self.in_b),
            psum.eq(self.in_psum),
            c_up.eq(self.in_c),
            c_up_valid.eq(self.in_c_valid),
        ]

        c = Signal(acc, name="c")  # OS accumulator
        w = Signal(signed(cfg.weight_bits), name="w")  # stationary operand
        stationary = self.in_control_dataflow

        product = Signal(signed(cfg.input_bits + cfg.weight_bits), name="product")
        m.d.comb += product.eq(a * Mux(stationary, w, b))
        total = Signal(acc, name="total")  # c after this product
        m.d.comb += total.eq(Mux(first, product, c + product))

        launch = Signal(name="launch")
        m.d.comb += launch.eq(~stationary & valid & last)

        m.d.comb += [
            self.out_a.eq(a),
            self.out_valid.eq(valid),
            self.out_first.eq(first),
            self.out_last.eq(last),
            self.out_b.eq(b),
            self.out_psum.eq(Mux(valid, psum + product, psum)),
            self.out_psum_valid.eq(valid),
            self.out_c.eq(Mux(launch, total, c_up)),
            self.out_c_valid.eq(launch | c_up_valid),
            self.out_w.eq(w),
            self.out_collision.eq(launch & c_up_valid),
        ]

        with m.If(~stationary & valid & ~last):
            m.d.sync += c.eq(total)
        with m.If(self.in_shift):
            m.d.sync += w.eq(self.in_w)

        return m
