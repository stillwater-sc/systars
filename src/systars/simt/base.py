"""
Base classes and protocols for SIMT/SIMD architecture implementations.

This module defines common interfaces that all architecture implementations
(NVIDIA, AMD, MasPar) should follow for interoperability.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ProcessorSim(Protocol):
    """
    Protocol for top-level processor simulation.

    All SIMT/SIMD processor implementations should conform to this interface
    to enable architecture-agnostic benchmarking and comparison.
    """

    def step(self) -> dict[str, Any]:
        """
        Execute one cycle of simulation.

        Returns:
            Dictionary with cycle status information including:
            - cycle: Current cycle number
            - state: Processor state
            - Any architecture-specific status
        """
        ...

    def run_to_completion(self, max_cycles: int = 10000) -> int:
        """
        Run until execution completes or max cycles reached.

        Args:
            max_cycles: Maximum cycles before timeout

        Returns:
            Number of cycles executed.
        """
        ...

    @property
    def done(self) -> bool:
        """Whether execution is complete."""
        ...

    @property
    def cycle(self) -> int:
        """Current cycle count."""
        ...

    def get_statistics(self) -> dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary with statistics including:
            - cycles: Total cycles
            - instructions: Total instructions executed
            - stalls: Total stall cycles
            - energy_pj: Total energy in picojoules (if available)
        """
        ...

    def get_energy_pj(self) -> float:
        """Get total energy consumed in picojoules."""
        ...


@runtime_checkable
class MemorySubsystem(Protocol):
    """
    Protocol for memory subsystem simulation.

    Defines basic read/write interface for memory models.
    """

    def read(self, address: int) -> int:
        """Read a 32-bit word from memory."""
        ...

    def write(self, address: int, value: int) -> None:
        """Write a 32-bit word to memory."""
        ...


class SIMTConfigBase(ABC):
    """
    Abstract base class for SIMT/SIMD configuration.

    Defines common configuration properties that all architectures share.
    """

    @property
    @abstractmethod
    def execution_width(self) -> int:
        """
        Number of parallel execution lanes.

        Examples:
        - NVIDIA: 32 (warp size)
        - AMD: 32 or 64 (wave size)
        - MasPar: 4096+ (PE array size)
        """
        ...

    @property
    @abstractmethod
    def local_memory_kb(self) -> int:
        """
        Size of fast local/shared memory in KB.

        Examples:
        - NVIDIA: 48 KB shared memory
        - AMD: 64 KB LDS
        - MasPar: 16 KB per PE × array size
        """
        ...

    @property
    @abstractmethod
    def num_compute_units(self) -> int:
        """
        Number of independent compute units.

        Examples:
        - NVIDIA: 4 partitions per SM
        - AMD: 2 CUs per WGP
        - MasPar: 1 (unified array)
        """
        ...


__all__ = [
    "ProcessorSim",
    "MemorySubsystem",
    "SIMTConfigBase",
]
