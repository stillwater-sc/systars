"""
ISA instructions for the systolic tile.

High-level instructions that tile an operation larger than the array and issue
the commands a host would issue to ``SystolicTop`` (LOAD, CONFIG, PRELOAD,
COMPUTE, STORE) on a shared command port (``command``), pushed ahead and ordered
by credits and tile-ready tokens (``controller/sync.py``):

    Matmul: C = A @ B
    Conv2d: Y = conv2d(X, F), as an implicit GEMM

Both run weight-stationary. Both check at START that the DMA can move every row
they load or store; see their modules for the layouts this allows today.
"""

from .command import CTRL_EXEC, CTRL_LOAD, CTRL_STORE, CommandIssuer, command_ports, connect_to_top
from .conv2d import Conv2d, Conv2dCmd, Conv2dError
from .matmul import Matmul, MatmulCmd, MatmulError

__all__ = [
    "CTRL_EXEC",
    "CTRL_LOAD",
    "CTRL_STORE",
    "CommandIssuer",
    "Conv2d",
    "Conv2dCmd",
    "Conv2dError",
    "Matmul",
    "MatmulCmd",
    "MatmulError",
    "command_ports",
    "connect_to_top",
]
