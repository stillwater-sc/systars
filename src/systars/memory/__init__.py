"""
Memory subsystem components.

This module contains the local memory components:
- LocalAddr: Address encoding/decoding utilities
- ScratchpadBank / Scratchpad: Multi-bank input/weight memory
- AccumulatorBank / Accumulator: Result memory with scale/activation
"""

from .accumulator import Accumulator, AccumulatorBank
from .local_addr import LocalAddr
from .scratchpad import Scratchpad, ScratchpadBank

__all__ = [
    "LocalAddr",
    "ScratchpadBank",
    "Scratchpad",
    "AccumulatorBank",
    "Accumulator",
]
