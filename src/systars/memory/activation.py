"""
Element-wise activation functions on packed accumulator rows.

An accumulator row packs ``n_elems`` signed elements of ``elem_bits`` each
(element 0 in the least-significant bits). Activations are applied to each
element independently. Comparing the packed row as one signed number would
test only the sign of the top element.
"""

from __future__ import annotations

from amaranth import Cat, Mux, Value

from ..config import Activation


def activate_row(row: Value, activation: Value, elem_bits: int, n_elems: int) -> Value:
    """Apply ``activation`` (an ``Activation`` code) to every element of ``row``.

    Supported: NONE, RELU (``max(0, x)``), RELU6 (``min(6, max(0, x))``, with 6
    unscaled, as in SYSTARS). Any other code passes data through unchanged.
    """
    lanes = []
    for i in range(n_elems):
        x = row[i * elem_bits : (i + 1) * elem_bits].as_signed()
        relu = Mux(x < 0, 0, x)
        relu6 = Mux(relu > 6, 6, relu)
        lanes.append(
            Mux(
                activation == Activation.RELU.value,
                relu,
                Mux(activation == Activation.RELU6.value, relu6, x),
            )[:elem_bits]
        )
    return Cat(*lanes)
