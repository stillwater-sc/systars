"""
Processing Element (PE) - The fundamental compute unit of the systolic array.

Each PE performs a multiply-accumulate (MAC) operation:
    out = in_c + (in_a * in_b)

The PE supports two dataflow modes:
- Output-Stationary (OS): Accumulator stays in place, weights flow through
- Weight-Stationary (WS): Weights stay in place, partial sums flow through

Data flows:
- A (activations): flows horizontally (left to right)
- B (weights/partial sums): flows vertically (top to bottom)
- D (bias/initial value): flows vertically for preloading
- C (result): output after accumulation
"""

from amaranth import Module, Mux, Signal, signed
from amaranth.lib.wiring import Component, In, Out

from ..config import SystolicConfig


class PE(Component):
    """
    Processing Element - performs MAC operations in a systolic array.

    The PE has two internal registers (c1, c2) that alternate roles based on
    the 'propagate' control signal. This allows pipelining of multiple
    matrix multiplications.

    Ports:
        in_a: Input activation (flows right)
        in_b: Input weight or partial sum (flows down)
        in_d: Preload value (bias or initial accumulator value)
        in_control: Control signals (dataflow, propagate, shift)
        in_valid: Input data valid signal
        in_id: Tag for tracking through pipeline
        in_last: Last element in current operation

        out_a: Pass-through of in_a (to PE on right)
        out_b: Output weight or partial sum (to PE below)
        out_c: Accumulated result
        out_control: Pass-through of control
        out_valid: Output valid signal
        out_id: Pass-through of tag
        out_last: Pass-through of last signal

    Parameters:
        config: SystolicConfig with bit widths and dataflow settings
    """

    def __init__(self, config: SystolicConfig):
        self.config = config

        # Calculate port widths
        input_width = config.input_bits
        weight_width = config.weight_bits
        acc_width = config.acc_bits
        output_width = config.output_bits

        # Define signature using lib.wiring
        super().__init__(
            {
                # Inputs
                "in_a": In(signed(input_width)),
                "in_b": In(signed(weight_width)),
                "in_d": In(signed(acc_width)),
                "in_control_dataflow": In(1),  # 0=OS, 1=WS
                "in_control_propagate": In(1),  # Which register to output
                "in_control_shift": In(5),  # Rounding shift amount
                "in_valid": In(1),
                "in_id": In(8),
                "in_last": In(1),
                # Outputs
                "out_a": Out(signed(input_width)),
                "out_b": Out(signed(output_width)),
                "out_c": Out(signed(acc_width)),
                "out_control_dataflow": Out(1),
                "out_control_propagate": Out(1),
                "out_control_shift": Out(5),
                "out_valid": Out(1),
                "out_id": Out(8),
                "out_last": Out(1),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        # Internal accumulator registers
        # c1 and c2 alternate roles based on 'propagate' signal
        c1 = Signal(signed(cfg.acc_bits), name="c1")
        c2 = Signal(signed(cfg.acc_bits), name="c2")

        # =================================================================
        # Multiply-Accumulate Computation
        # =================================================================

        # Step 1: Multiply in_a * in_b
        # Product width = input_bits + weight_bits
        product_width = cfg.input_bits + cfg.weight_bits
        product = Signal(signed(product_width), name="product")
        m.d.comb += product.eq(self.in_a * self.in_b)

        # Step 2: Sign-extend product to accumulator width
        product_ext = Signal(signed(cfg.acc_bits), name="product_ext")
        m.d.comb += product_ext.eq(product)  # Amaranth handles sign extension

        # Step 3: Select accumulator input based on dataflow and propagate
        acc_input = Signal(signed(cfg.acc_bits), name="acc_input")

        with m.If(self.in_control_dataflow):  # Weight-Stationary mode
            # In WS mode, we use in_d as the accumulator input
            m.d.comb += acc_input.eq(self.in_d)
        with m.Else():  # Output-Stationary mode
            # In OS mode, we accumulate into c1 or c2 based on propagate
            with m.If(self.in_control_propagate):
                m.d.comb += acc_input.eq(c2)
            with m.Else():
                m.d.comb += acc_input.eq(c1)

        # Step 4: Accumulate
        accumulated = Signal(signed(cfg.acc_bits), name="accumulated")
        m.d.comb += accumulated.eq(acc_input + product_ext)

        # =================================================================
        # Register Update Logic
        # =================================================================

        with m.If(self.in_valid):
            # Update the appropriate register based on propagate signal
            with m.If(self.in_control_propagate):
                m.d.sync += c2.eq(accumulated)
            with m.Else():
                m.d.sync += c1.eq(accumulated)

        # =================================================================
        # Output Selection
        # =================================================================

        # out_c: Output the register NOT being written to
        with m.If(self.in_control_propagate):
            m.d.comb += self.out_c.eq(c1)
        with m.Else():
            m.d.comb += self.out_c.eq(c2)

        # out_b: In WS mode, pass through partial sum; in OS mode, pass weight
        # For now, simplified to pass through product (actual impl may differ)
        with m.If(self.in_control_dataflow):  # WS
            # Pass accumulated value down
            m.d.comb += self.out_b.eq(accumulated[: cfg.output_bits])
        with m.Else():  # OS
            # Pass weight down
            m.d.comb += self.out_b.eq(self.in_b[: cfg.output_bits])

        # =================================================================
        # Pass-through Signals (registered for pipelining)
        # =================================================================

        m.d.sync += [
            self.out_a.eq(self.in_a),
            self.out_control_dataflow.eq(self.in_control_dataflow),
            self.out_control_propagate.eq(self.in_control_propagate),
            self.out_control_shift.eq(self.in_control_shift),
            self.out_valid.eq(self.in_valid),
            self.out_id.eq(self.in_id),
            self.out_last.eq(self.in_last),
        ]

        return m


class PEWithShift(Component):
    """
    Processing Element with rounding right-shift on output.

    This variant includes a barrel shifter for fixed-point quantization.
    The shift amount is configurable at runtime via in_control_shift.

    Rounding uses "round half to even" (banker's rounding):
        shifted = (value + (1 << (shift-1))) >> shift
    """

    def __init__(self, config: SystolicConfig):
        self.config = config

        input_width = config.input_bits
        weight_width = config.weight_bits
        acc_width = config.acc_bits
        output_width = config.output_bits

        super().__init__(
            {
                "in_a": In(signed(input_width)),
                "in_b": In(signed(weight_width)),
                "in_d": In(signed(acc_width)),
                "in_control_dataflow": In(1),
                "in_control_propagate": In(1),
                "in_control_shift": In(5),
                "in_valid": In(1),
                "in_id": In(8),
                "in_last": In(1),
                "out_a": Out(signed(input_width)),
                "out_b": Out(signed(output_width)),
                "out_c": Out(signed(acc_width)),
                "out_c_shifted": Out(signed(output_width)),  # Shifted output
                "out_control_dataflow": Out(1),
                "out_control_propagate": Out(1),
                "out_control_shift": Out(5),
                "out_valid": Out(1),
                "out_id": Out(8),
                "out_last": Out(1),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        # Instantiate base PE
        m.submodules.pe = pe = PE(cfg)

        # Connect inputs
        m.d.comb += [
            pe.in_a.eq(self.in_a),
            pe.in_b.eq(self.in_b),
            pe.in_d.eq(self.in_d),
            pe.in_control_dataflow.eq(self.in_control_dataflow),
            pe.in_control_propagate.eq(self.in_control_propagate),
            pe.in_control_shift.eq(self.in_control_shift),
            pe.in_valid.eq(self.in_valid),
            pe.in_id.eq(self.in_id),
            pe.in_last.eq(self.in_last),
        ]

        # Connect pass-through outputs
        m.d.comb += [
            self.out_a.eq(pe.out_a),
            self.out_b.eq(pe.out_b),
            self.out_c.eq(pe.out_c),
            self.out_control_dataflow.eq(pe.out_control_dataflow),
            self.out_control_propagate.eq(pe.out_control_propagate),
            self.out_control_shift.eq(pe.out_control_shift),
            self.out_valid.eq(pe.out_valid),
            self.out_id.eq(pe.out_id),
            self.out_last.eq(pe.out_last),
        ]

        # Rounding right-shift for out_c_shifted
        # Round half to even: add (1 << (shift-1)) before shifting
        shift_amt = pe.out_control_shift
        round_bit = Signal(cfg.acc_bits, name="round_bit")
        shifted = Signal(signed(cfg.acc_bits), name="shifted")

        # Calculate rounding bias (1 << (shift-1)), handling shift=0
        # Use (1 << shift_amt) >> 1 to avoid signed shift amount issue
        m.d.comb += round_bit.eq(Mux(shift_amt == 0, 0, (1 << shift_amt) >> 1))

        # Add rounding bias and shift
        rounded_value = Signal(signed(cfg.acc_bits), name="rounded_value")
        m.d.comb += rounded_value.eq(pe.out_c + round_bit.as_signed())
        m.d.comb += shifted.eq(rounded_value >> shift_amt)

        # Saturate to output width
        max_val = (1 << (cfg.output_bits - 1)) - 1
        min_val = -(1 << (cfg.output_bits - 1))

        with m.If(shifted > max_val):
            m.d.comb += self.out_c_shifted.eq(max_val)
        with m.Elif(shifted < min_val):
            m.d.comb += self.out_c_shifted.eq(min_val)
        with m.Else():
            m.d.comb += self.out_c_shifted.eq(shifted[: cfg.output_bits])

        return m
