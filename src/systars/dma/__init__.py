"""
DMA engines for external memory transfers.

This module contains:
- StreamReader: External memory -> local memory (read DMA)
- StreamWriter: Local memory -> external memory (write DMA)
- StridedStreamReader: DMA with strided access for matrix transpose
- DescriptorEngine: Hardware engine for executing descriptor chains
"""

from .descriptor_engine import DescriptorEngine
from .reader import StreamReader
from .strided_reader import StridedStreamReader
from .writer import StreamWriter

__all__ = ["DescriptorEngine", "StreamReader", "StridedStreamReader", "StreamWriter"]
