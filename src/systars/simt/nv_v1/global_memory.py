"""
Global Memory Simulation for SIMT Streaming Multiprocessor.

Provides a simple dict-based memory model for functional verification,
following GPU global memory semantics (high latency, coalesced access).

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    GLOBAL MEMORY (DRAM)                              │
    │                                                                      │
    │  • 32-bit addressable (byte addresses)                              │
    │  • Variable latency (100-400 cycles depending on access pattern)    │
    │  • Coalescing: 128B transactions for adjacent accesses              │
    │  • High energy cost per access (~200pJ per transaction)             │
    │                                                                      │
    │  ┌──────────────────────────────────────────────────────────────┐   │
    │  │                    ADDRESS SPACE                              │   │
    │  │  0x0000_0000 - 0x3FFF_FFFF (1GB, global memory region)       │   │
    │  │                                                               │   │
    │  │  Matrix storage layout (row-major):                          │   │
    │  │    A[M,K] at base_a: A[i,k] = base_a + (i*K + k)*4           │   │
    │  │    B[K,N] at base_b: B[k,j] = base_b + (k*N + j)*4           │   │
    │  │    C[M,N] at base_c: C[i,j] = base_c + (i*N + j)*4           │   │
    │  └──────────────────────────────────────────────────────────────┘   │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import SIMTConfig


@dataclass
class GlobalMemorySim:
    """
    Behavioral simulation model for global memory (DRAM).

    Provides:
    - Simple dict-based storage for functional correctness
    - Matrix read/write for GEMM verification
    - Energy tracking per access
    - Statistics for coalescing analysis
    """

    config: SIMTConfig

    # Storage: dict mapping byte address -> 32-bit value
    storage: dict[int, int] = field(default_factory=dict)

    # Statistics
    total_reads: int = 0
    total_writes: int = 0
    total_bytes_read: int = 0
    total_bytes_written: int = 0
    total_energy_pj: float = 0.0

    def read(self, addr: int) -> int:
        """
        Read a 32-bit word from memory.

        Args:
            addr: Byte address (should be 4-byte aligned)

        Returns:
            32-bit value at address (0 if uninitialized)
        """
        # Align to 4-byte boundary
        aligned_addr = addr & ~0x3
        value = self.storage.get(aligned_addr, 0)
        self.total_reads += 1
        self.total_bytes_read += 4
        self.total_energy_pj += self.config.dram_access_energy_pj
        return value

    def write(self, addr: int, value: int) -> None:
        """
        Write a 32-bit word to memory.

        Args:
            addr: Byte address (should be 4-byte aligned)
            value: 32-bit value to write
        """
        # Align to 4-byte boundary
        aligned_addr = addr & ~0x3
        self.storage[aligned_addr] = value & 0xFFFF_FFFF
        self.total_writes += 1
        self.total_bytes_written += 4
        self.total_energy_pj += self.config.dram_access_energy_pj

    def read_bulk(self, base_addr: int, count: int) -> list[int]:
        """
        Read multiple consecutive 32-bit words.

        Args:
            base_addr: Starting byte address
            count: Number of 32-bit words to read

        Returns:
            List of values
        """
        return [self.read(base_addr + i * 4) for i in range(count)]

    def write_bulk(self, base_addr: int, values: list[int]) -> None:
        """
        Write multiple consecutive 32-bit words.

        Args:
            base_addr: Starting byte address
            values: List of 32-bit values to write
        """
        for i, value in enumerate(values):
            self.write(base_addr + i * 4, value)

    # =========================================================================
    # Matrix Operations for GEMM Testing
    # =========================================================================

    def load_matrix(
        self,
        base_addr: int,
        matrix: np.ndarray,
        dtype: type = np.int32,
    ) -> None:
        """
        Load a NumPy matrix into global memory (row-major order).

        Args:
            base_addr: Base address for matrix storage
            matrix: 2D NumPy array
            dtype: Data type (int32 or float32)
        """
        rows, cols = matrix.shape
        flat: np.ndarray = matrix.astype(dtype).flatten()

        for idx, val in enumerate(flat):
            addr = base_addr + idx * 4
            if dtype == np.float32:
                # Store float32 as its bit representation
                self.storage[addr] = int(np.float32(val).view(np.uint32))
            else:
                self.storage[addr] = int(val) & 0xFFFF_FFFF

    def read_matrix(
        self,
        base_addr: int,
        rows: int,
        cols: int,
        dtype: type = np.int32,
    ) -> np.ndarray:
        """
        Read a matrix from global memory (row-major order).

        Args:
            base_addr: Base address of matrix
            rows: Number of rows
            cols: Number of columns
            dtype: Data type (int32 or float32)

        Returns:
            2D NumPy array
        """
        result: np.ndarray = np.zeros((rows, cols), dtype=dtype)

        for i in range(rows):
            for j in range(cols):
                addr = base_addr + (i * cols + j) * 4
                value = self.storage.get(addr, 0)

                if dtype == np.float32:
                    # Interpret bits as float32
                    result[i, j] = np.uint32(value).view(np.float32)
                else:
                    result[i, j] = value

        return result

    def matrix_address(self, base_addr: int, row: int, col: int, stride: int) -> int:
        """
        Compute address for matrix element (row-major).

        Args:
            base_addr: Matrix base address
            row: Row index
            col: Column index
            stride: Number of columns (row stride)

        Returns:
            Byte address of element
        """
        return base_addr + (row * stride + col) * 4

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def clear(self) -> None:
        """Clear all memory contents."""
        self.storage.clear()

    def reset_statistics(self) -> None:
        """Reset access statistics."""
        self.total_reads = 0
        self.total_writes = 0
        self.total_bytes_read = 0
        self.total_bytes_written = 0
        self.total_energy_pj = 0.0

    def get_statistics(self) -> dict[str, Any]:
        """Get memory access statistics."""
        return {
            "total_reads": self.total_reads,
            "total_writes": self.total_writes,
            "total_bytes_read": self.total_bytes_read,
            "total_bytes_written": self.total_bytes_written,
            "total_energy_pj": self.total_energy_pj,
            "storage_entries": len(self.storage),
        }

    def hexdump(self, start_addr: int, length: int = 64) -> str:
        """
        Generate hex dump of memory region.

        Args:
            start_addr: Starting address
            length: Number of bytes to dump

        Returns:
            Formatted hex dump string
        """
        lines = []
        for offset in range(0, length, 16):
            addr = start_addr + offset
            words = []
            for w in range(0, 16, 4):
                word_addr = addr + w
                value = self.storage.get(word_addr & ~0x3, 0)
                words.append(f"{value:08X}")

            line = f"{addr:08X}: {' '.join(words)}"
            lines.append(line)

        return "\n".join(lines)
