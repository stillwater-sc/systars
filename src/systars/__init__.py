"""
Systars - A Python-based systolic array RTL generator.

This package provides configurable hardware generation for systolic array
accelerators using Amaranth HDL.
"""

from .config import Activation, Dataflow, SystolicConfig

__version__ = "0.1.0"
__all__ = ["SystolicConfig", "Dataflow", "Activation", "__version__"]
