"""
Simulated DRAM Model for Systars Examples.

This module provides a software model of external DRAM memory that the
systolic array accelerator interacts with. It supports:
- Loading input tensors (activations, weights) into memory
- Retrieving output tensors (results) from memory
- Bus-width aligned transfers matching hardware behavior

In a real system, this would be replaced by actual memory-mapped I/O
or a DMA driver interface.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MemoryRegion:
    """Describes a named region in DRAM."""

    name: str
    base_addr: int
    size_bytes: int
    element_bits: int = 8
    shape: tuple[int, ...] = ()


@dataclass
class SimulatedDRAM:
    """
    Simulated DRAM for systolic array demonstrations.

    This class models external memory that the accelerator accesses via DMA.
    It provides methods to:
    - Allocate regions for matrices
    - Load numpy arrays into memory
    - Read results back as numpy arrays

    Attributes:
        size_bytes: Total DRAM capacity in bytes
        buswidth: Bus width in bits (must match hardware config)
        data: Raw byte storage
        regions: Named memory regions for tracking allocations

    Example:
        >>> dram = SimulatedDRAM(size_bytes=1024*1024)
        >>> A = np.array([[1, 2], [3, 4]], dtype=np.int8)
        >>> dram.store_matrix("A", 0x1000, A)
        >>> A_back = dram.load_matrix("A", 0x1000, (2, 2), np.int8)
        >>> np.testing.assert_array_equal(A, A_back)
    """

    size_bytes: int = 1024 * 1024  # 1 MB default
    buswidth: int = 128  # bits
    data: bytearray = field(default_factory=lambda: bytearray())
    regions: dict = field(default_factory=dict)

    def __post_init__(self):
        """Initialize the data array after dataclass init."""
        if len(self.data) == 0:
            self.data = bytearray(self.size_bytes)
        self.bytes_per_beat = self.buswidth // 8

    def allocate_region(
        self,
        name: str,
        base_addr: int,
        shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> MemoryRegion:
        """
        Allocate a named memory region for a tensor.

        Args:
            name: Identifier for this region
            base_addr: Starting address in DRAM
            shape: Tensor dimensions
            dtype: NumPy dtype (np.int8, np.int32, etc.)

        Returns:
            MemoryRegion descriptor

        Raises:
            ValueError: If region overlaps existing allocation or exceeds DRAM
        """
        element_bits = dtype.itemsize * 8
        total_elements = int(np.prod(shape))
        size_bytes = total_elements * (element_bits // 8)

        if base_addr + size_bytes > self.size_bytes:
            raise ValueError(
                f"Region '{name}' at 0x{base_addr:X} with size {size_bytes} "
                f"exceeds DRAM capacity {self.size_bytes}"
            )

        region = MemoryRegion(
            name=name,
            base_addr=base_addr,
            size_bytes=size_bytes,
            element_bits=element_bits,
            shape=shape,
        )
        self.regions[name] = region
        return region

    def store_matrix(
        self,
        name: str,
        base_addr: int,
        matrix: np.ndarray,
    ) -> MemoryRegion:
        """
        Store a numpy matrix into DRAM.

        Args:
            name: Identifier for this data
            base_addr: Starting address in DRAM
            matrix: NumPy array to store

        Returns:
            MemoryRegion descriptor for the stored data
        """
        region = self.allocate_region(name, base_addr, matrix.shape, matrix.dtype)

        flat = matrix.flatten()
        bytes_per_elem = matrix.dtype.itemsize

        for i, val in enumerate(flat):
            addr = base_addr + i * bytes_per_elem
            val = int(val)

            # Handle signed values - convert to unsigned representation
            if val < 0:
                val = val + (1 << (bytes_per_elem * 8))

            # Store little-endian
            for b in range(bytes_per_elem):
                self.data[addr + b] = (val >> (b * 8)) & 0xFF

        return region

    def load_matrix(
        self,
        name: str,  # noqa: ARG002 - kept for API consistency with store_matrix
        base_addr: int,
        shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> np.ndarray:
        """
        Load a matrix from DRAM as a numpy array.

        Args:
            name: Identifier (for tracking, matches store_matrix API)
            base_addr: Starting address in DRAM
            shape: Expected matrix dimensions
            dtype: NumPy dtype to interpret data as

        Returns:
            NumPy array with the loaded data
        """
        bytes_per_elem = np.dtype(dtype).itemsize
        element_bits = bytes_per_elem * 8
        signed = np.issubdtype(dtype, np.signedinteger)

        result = np.zeros(shape, dtype=dtype)
        flat_result = result.flatten()

        for i in range(len(flat_result)):
            addr = base_addr + i * bytes_per_elem
            val = 0

            # Load little-endian
            for b in range(bytes_per_elem):
                val |= self.data[addr + b] << (b * 8)

            # Sign extend if needed
            if signed and (val & (1 << (element_bits - 1))):
                val -= 1 << element_bits

            flat_result[i] = val

        return flat_result.reshape(shape)

    def read_beat(self, addr: int) -> int:
        """Read one bus-width beat from DRAM."""
        val = 0
        for b in range(self.bytes_per_beat):
            if addr + b < self.size_bytes:
                val |= self.data[addr + b] << (b * 8)
        return val

    def write_beat(self, addr: int, data: int, strb: int | None = None):
        """Write one bus-width beat to DRAM with optional byte strobes."""
        if strb is None:
            strb = (1 << self.bytes_per_beat) - 1  # All bytes valid
        for b in range(self.bytes_per_beat):
            if ((strb >> b) & 1) and (addr + b < self.size_bytes):
                self.data[addr + b] = (data >> (b * 8)) & 0xFF

    def dump_region(self, name: str, num_bytes: int = 64) -> str:
        """
        Dump a memory region as hex for debugging.

        Args:
            name: Region name
            num_bytes: Maximum bytes to dump

        Returns:
            Hex dump string
        """
        if name not in self.regions:
            return f"Region '{name}' not found"

        region = self.regions[name]
        addr = region.base_addr
        lines = [f"Region '{name}' at 0x{addr:08X} ({region.size_bytes} bytes):"]

        for offset in range(0, min(num_bytes, region.size_bytes), 16):
            hex_bytes = " ".join(
                f"{self.data[addr + offset + i]:02X}"
                for i in range(min(16, region.size_bytes - offset))
            )
            lines.append(f"  0x{addr + offset:08X}: {hex_bytes}")

        if region.size_bytes > num_bytes:
            lines.append(f"  ... ({region.size_bytes - num_bytes} more bytes)")

        return "\n".join(lines)

    def print_memory_map(self):
        """Print the current memory allocation map."""
        print("\nMemory Map:")
        print("-" * 60)
        for name, region in sorted(self.regions.items(), key=lambda x: x[1].base_addr):
            end_addr = region.base_addr + region.size_bytes - 1
            print(
                f"  {name:20s}: 0x{region.base_addr:08X} - 0x{end_addr:08X} "
                f"({region.size_bytes:6d} bytes) shape={region.shape}"
            )
        print("-" * 60)
