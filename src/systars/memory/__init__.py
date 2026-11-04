"""
Memory subsystem components.

This module contains the local memory components:
- LocalAddr: Address encoding/decoding utilities
- ScratchpadBank / Scratchpad: Multi-bank input/weight memory
- AccumulatorBank / Accumulator: Result memory with scale/activation
- SkewBuffer / SkewedArrayFeeder: Input skew buffers for systolic feeding
- StreamBuffers / ResultBuffers: L1, the stream buffers that feed the fabric and the
  result buffers that drain it
"""

from .accumulator import Accumulator, AccumulatorBank
from .local_addr import LocalAddr
from .scratchpad import Scratchpad, ScratchpadBank
from .skew_buffer import SkewBuffer, SkewedArrayFeeder, SRAMReadScheduler
from .stream_buffer import ResultBuffers, StreamBuffers

__all__ = [
    "LocalAddr",
    "ScratchpadBank",
    "Scratchpad",
    "AccumulatorBank",
    "Accumulator",
    "SkewBuffer",
    "SkewedArrayFeeder",
    "SRAMReadScheduler",
    "StreamBuffers",
    "ResultBuffers",
]
