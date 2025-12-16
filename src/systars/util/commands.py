"""
Command and descriptor definitions for the systars accelerator.

This module defines:
1. DMA opcodes for descriptor-based memory operations
2. Execute controller opcodes for compute operations
3. Descriptor format for chained DMA operations
4. Helper functions for creating descriptors

Descriptor Format (64 bytes, 8-byte aligned):
    Offset  Size  Field
    0x00    1     opcode      - Operation code (MEMCPY, FILL, etc.)
    0x01    1     flags       - Operation flags
    0x02    2     reserved
    0x04    4     length      - Transfer length in bytes
    0x08    8     src_addr    - Source address (or fill pattern)
    0x10    8     dst_addr    - Destination address
    0x18    8     next_desc   - Pointer to next descriptor (0 = end of chain)
    0x20    8     completion  - Completion status writeback address
    0x28    8     user_data   - User-defined data (passed to completion)
    0x30    8     reserved2
    0x38    8     reserved3

Memory Map (physical addresses):
    0x0000_0000_0000 - 0x0000_FFFF_FFFF : External DRAM (via DMA)
    0x0001_0000_0000 - 0x0001_0000_FFFF : Scratchpad (local)
    0x0002_0000_0000 - 0x0002_0000_FFFF : Accumulator (local)
"""

from dataclasses import dataclass
from enum import IntEnum, IntFlag

# =============================================================================
# DMA Opcodes (descriptor-based operations)
# =============================================================================


class DmaOpcode(IntEnum):
    """
    DMA operation codes for descriptor-based memory transfers.

    These are operation-oriented (what to do) rather than direction-oriented.
    The source and destination addresses determine the actual data path.
    """

    # Data movement operations
    MEMCPY = 0x00  # Copy src -> dst
    FILL = 0x01  # Fill dst with pattern from src field
    GATHER = 0x02  # Gather scattered elements to contiguous dst
    SCATTER = 0x03  # Scatter contiguous src to scattered locations

    # Data integrity operations
    CRC32 = 0x10  # Compute CRC32 over src region
    CHECKSUM = 0x11  # Compute simple checksum

    # Synchronization operations
    FENCE = 0x20  # Memory barrier - wait for all prior ops
    NOP = 0x21  # No operation (useful for padding/alignment)

    # Completion operations
    SIGNAL = 0x30  # Write completion value to address
    INTERRUPT = 0x31  # Generate interrupt on completion


class DmaFlags(IntFlag):
    """Flags for DMA descriptor operations."""

    NONE = 0x00
    CHAIN = 0x01  # Continue to next descriptor
    INTERRUPT_ON_COMPLETE = 0x02  # Generate interrupt when done
    WRITEBACK_STATUS = 0x04  # Write status to completion address
    SRC_IS_PATTERN = 0x08  # src_addr is fill pattern, not address
    DST_IS_LOCAL = 0x10  # Destination is local memory (skip DMA)
    SRC_IS_LOCAL = 0x20  # Source is local memory (skip DMA)


# =============================================================================
# Execute Controller Opcodes (compute operations)
# =============================================================================


class ExecOpcode(IntEnum):
    """
    Execute controller opcodes for systolic array operations.

    These control the compute fabric, not memory movement.
    """

    # Configuration
    CONFIG = 0x00  # Configure dataflow mode, shift amount
    CONFIG_EX = 0x00  # Legacy alias for CONFIG

    # Compute operations
    PRELOAD = 0x10  # Preload bias/initial values from accumulator
    COMPUTE = 0x11  # Execute matmul: A @ B + C
    COMPUTE_PRELOAD = 0x12  # Fused preload + compute
    COMPUTE_ACCUMULATE = 0x13  # Compute and accumulate to existing result


class ConfigFlags(IntEnum):
    """Configuration flags for execute controller."""

    # Dataflow mode (2 bits)
    DATAFLOW_OS = 0x0000  # Output-stationary
    DATAFLOW_WS = 0x0001  # Weight-stationary
    DATAFLOW_IS = 0x0002  # Input-stationary

    # Accumulate mode
    ACCUMULATE = 0x0010  # Accumulate results (vs overwrite)

    # Activation function (4 bits)
    ACT_NONE = 0x0000
    ACT_RELU = 0x0100
    ACT_RELU6 = 0x0200
    ACT_SIGMOID = 0x0300


# =============================================================================
# Memory Map Constants
# =============================================================================


class MemoryRegion(IntEnum):
    """Base addresses for memory regions in the unified address space."""

    DRAM_BASE = 0x0000_0000_0000
    DRAM_SIZE = 0x0001_0000_0000  # 4GB DRAM region

    SCRATCHPAD_BASE = 0x0001_0000_0000
    SCRATCHPAD_SIZE = 0x0000_0001_0000  # 64KB scratchpad

    ACCUMULATOR_BASE = 0x0002_0000_0000
    ACCUMULATOR_SIZE = 0x0000_0001_0000  # 64KB accumulator

    # Future expansion
    RESERVED_BASE = 0x0003_0000_0000


def get_memory_region(addr: int) -> str:
    """Determine which memory region an address belongs to."""
    if addr < MemoryRegion.SCRATCHPAD_BASE:
        return "DRAM"
    elif addr < MemoryRegion.ACCUMULATOR_BASE:
        return "SCRATCHPAD"
    elif addr < MemoryRegion.RESERVED_BASE:
        return "ACCUMULATOR"
    else:
        return "RESERVED"


def is_local_memory(addr: int) -> bool:
    """Check if address is in local memory (scratchpad or accumulator)."""
    return addr >= MemoryRegion.SCRATCHPAD_BASE


# =============================================================================
# Descriptor Data Structure
# =============================================================================


@dataclass
class DmaDescriptor:
    """
    DMA descriptor for chained memory operations.

    This is a software representation of the hardware descriptor format.
    Use to_bytes() to convert to the 64-byte hardware format.
    """

    opcode: DmaOpcode
    flags: DmaFlags = DmaFlags.NONE
    length: int = 0
    src_addr: int = 0
    dst_addr: int = 0
    next_desc: int = 0  # 0 = end of chain
    completion: int = 0  # Status writeback address
    user_data: int = 0  # Passed to completion handler

    def to_bytes(self) -> bytes:
        """Convert descriptor to 64-byte hardware format (little-endian)."""
        import struct

        return struct.pack(
            "<BBHIQQQQQQQ",  # 64 bytes total
            self.opcode,
            self.flags,
            0,  # reserved
            self.length,
            self.src_addr,
            self.dst_addr,
            self.next_desc,
            self.completion,
            self.user_data,
            0,  # reserved2
            0,  # reserved3
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "DmaDescriptor":
        """Parse descriptor from 64-byte hardware format."""
        import struct

        (
            opcode,
            flags,
            _reserved,
            length,
            src_addr,
            dst_addr,
            next_desc,
            completion,
            user_data,
            _reserved2,
            _reserved3,
        ) = struct.unpack("<BBHIQQQQQQQ", data[:64])

        return cls(
            opcode=DmaOpcode(opcode),
            flags=DmaFlags(flags),
            length=length,
            src_addr=src_addr,
            dst_addr=dst_addr,
            next_desc=next_desc,
            completion=completion,
            user_data=user_data,
        )

    def __repr__(self) -> str:
        return (
            f"DmaDescriptor({self.opcode.name}, "
            f"src=0x{self.src_addr:012x}, dst=0x{self.dst_addr:012x}, "
            f"len={self.length}, next=0x{self.next_desc:x})"
        )


# =============================================================================
# Descriptor Builder Functions
# =============================================================================


def make_memcpy(
    src_addr: int,
    dst_addr: int,
    length: int,
    next_desc: int = 0,
    completion: int = 0,
    interrupt: bool = False,
) -> DmaDescriptor:
    """
    Create a MEMCPY descriptor.

    Args:
        src_addr: Source address (DRAM, scratchpad, or accumulator)
        dst_addr: Destination address
        length: Transfer length in bytes
        next_desc: Pointer to next descriptor (0 = end of chain)
        completion: Address for status writeback
        interrupt: Generate interrupt on completion

    Returns:
        DmaDescriptor configured for MEMCPY operation
    """
    flags = DmaFlags.NONE
    if next_desc != 0:
        flags |= DmaFlags.CHAIN
    if interrupt:
        flags |= DmaFlags.INTERRUPT_ON_COMPLETE
    if completion != 0:
        flags |= DmaFlags.WRITEBACK_STATUS
    if is_local_memory(src_addr):
        flags |= DmaFlags.SRC_IS_LOCAL
    if is_local_memory(dst_addr):
        flags |= DmaFlags.DST_IS_LOCAL

    return DmaDescriptor(
        opcode=DmaOpcode.MEMCPY,
        flags=flags,
        length=length,
        src_addr=src_addr,
        dst_addr=dst_addr,
        next_desc=next_desc,
        completion=completion,
    )


def make_fill(
    dst_addr: int,
    length: int,
    pattern: int = 0,
    next_desc: int = 0,
    completion: int = 0,
    interrupt: bool = False,
) -> DmaDescriptor:
    """
    Create a FILL descriptor to fill memory with a pattern.

    Args:
        dst_addr: Destination address
        length: Region length in bytes
        pattern: 64-bit fill pattern
        next_desc: Pointer to next descriptor
        completion: Address for status writeback
        interrupt: Generate interrupt on completion

    Returns:
        DmaDescriptor configured for FILL operation
    """
    flags = DmaFlags.SRC_IS_PATTERN
    if next_desc != 0:
        flags |= DmaFlags.CHAIN
    if interrupt:
        flags |= DmaFlags.INTERRUPT_ON_COMPLETE
    if completion != 0:
        flags |= DmaFlags.WRITEBACK_STATUS
    if is_local_memory(dst_addr):
        flags |= DmaFlags.DST_IS_LOCAL

    return DmaDescriptor(
        opcode=DmaOpcode.FILL,
        flags=flags,
        length=length,
        src_addr=pattern,  # Pattern stored in src_addr field
        dst_addr=dst_addr,
        next_desc=next_desc,
        completion=completion,
    )


def make_fence(
    next_desc: int = 0,
    completion: int = 0,
    interrupt: bool = False,
) -> DmaDescriptor:
    """
    Create a FENCE descriptor (memory barrier).

    Ensures all prior DMA operations complete before continuing.

    Args:
        next_desc: Pointer to next descriptor
        completion: Address for status writeback
        interrupt: Generate interrupt on completion

    Returns:
        DmaDescriptor configured for FENCE operation
    """
    flags = DmaFlags.NONE
    if next_desc != 0:
        flags |= DmaFlags.CHAIN
    if interrupt:
        flags |= DmaFlags.INTERRUPT_ON_COMPLETE
    if completion != 0:
        flags |= DmaFlags.WRITEBACK_STATUS

    return DmaDescriptor(
        opcode=DmaOpcode.FENCE,
        flags=flags,
        next_desc=next_desc,
        completion=completion,
    )


def make_crc32(
    src_addr: int,
    length: int,
    result_addr: int,
    next_desc: int = 0,
    interrupt: bool = False,
) -> DmaDescriptor:
    """
    Create a CRC32 descriptor.

    Computes CRC32 over source region and writes result to destination.

    Args:
        src_addr: Source address to compute CRC over
        length: Region length in bytes
        result_addr: Address to write 32-bit CRC result
        next_desc: Pointer to next descriptor
        interrupt: Generate interrupt on completion

    Returns:
        DmaDescriptor configured for CRC32 operation
    """
    flags = DmaFlags.NONE
    if next_desc != 0:
        flags |= DmaFlags.CHAIN
    if interrupt:
        flags |= DmaFlags.INTERRUPT_ON_COMPLETE
    if is_local_memory(src_addr):
        flags |= DmaFlags.SRC_IS_LOCAL

    return DmaDescriptor(
        opcode=DmaOpcode.CRC32,
        flags=flags,
        length=length,
        src_addr=src_addr,
        dst_addr=result_addr,
        next_desc=next_desc,
        completion=result_addr,  # CRC written to completion address
    )


# =============================================================================
# Descriptor Chain Builder
# =============================================================================


class DescriptorChain:
    """
    Builder for creating chains of DMA descriptors.

    Example:
        chain = DescriptorChain(base_addr=0x1000)
        chain.memcpy(src=0x0, dst=SCRATCHPAD_BASE, length=4096)
        chain.memcpy(src=0x1000, dst=SCRATCHPAD_BASE + 4096, length=4096)
        chain.fence()
        descriptors = chain.build()
    """

    def __init__(self, base_addr: int = 0):
        """
        Initialize descriptor chain builder.

        Args:
            base_addr: Base address where descriptors will be stored in memory
        """
        self.base_addr = base_addr
        self.descriptors: list[DmaDescriptor] = []

    def memcpy(self, src: int, dst: int, length: int, interrupt: bool = False) -> "DescriptorChain":
        """Add MEMCPY operation to chain."""
        desc = make_memcpy(src, dst, length, interrupt=interrupt)
        self.descriptors.append(desc)
        return self

    def fill(
        self, dst: int, length: int, pattern: int = 0, interrupt: bool = False
    ) -> "DescriptorChain":
        """Add FILL operation to chain."""
        desc = make_fill(dst, length, pattern, interrupt=interrupt)
        self.descriptors.append(desc)
        return self

    def fence(self, interrupt: bool = False) -> "DescriptorChain":
        """Add FENCE operation to chain."""
        desc = make_fence(interrupt=interrupt)
        self.descriptors.append(desc)
        return self

    def crc32(
        self, src: int, length: int, result: int, interrupt: bool = False
    ) -> "DescriptorChain":
        """Add CRC32 operation to chain."""
        desc = make_crc32(src, length, result, interrupt=interrupt)
        self.descriptors.append(desc)
        return self

    def build(self, completion_addr: int = 0) -> list[DmaDescriptor]:
        """
        Finalize the chain by linking descriptors.

        Args:
            completion_addr: Address for final completion writeback

        Returns:
            List of linked DmaDescriptor objects
        """
        if not self.descriptors:
            return []

        # Link descriptors: each points to the next
        desc_size = 64  # bytes per descriptor
        for i, desc in enumerate(self.descriptors[:-1]):
            desc.next_desc = self.base_addr + (i + 1) * desc_size
            desc.flags |= DmaFlags.CHAIN

        # Last descriptor ends the chain
        last = self.descriptors[-1]
        last.next_desc = 0
        last.flags &= ~DmaFlags.CHAIN
        if completion_addr:
            last.completion = completion_addr
            last.flags |= DmaFlags.WRITEBACK_STATUS

        return self.descriptors

    def to_bytes(self) -> bytes:
        """Convert entire chain to bytes for writing to memory."""
        return b"".join(desc.to_bytes() for desc in self.descriptors)


# =============================================================================
# Legacy Command Support (for ExecuteController)
# =============================================================================

# Keep OpCode as alias for backward compatibility during transition
OpCode = ExecOpcode


def make_config_ex(
    dataflow: int = 0,
    shift: int = 0,
    id_tag: int = 0,
) -> int:
    """Create CONFIG command for execute controller."""
    flags = dataflow & 0x0003
    data = shift & 0x1F
    return (ExecOpcode.CONFIG << 56) | (id_tag << 48) | (flags << 32) | data


def make_preload(
    acc_addr: int,
    rows: int = 1,
    id_tag: int = 0,
) -> int:
    """Create PRELOAD command for execute controller."""
    flags = rows & 0xFFFF
    return (ExecOpcode.PRELOAD << 56) | (id_tag << 48) | (flags << 32) | acc_addr


def make_compute(
    a_addr: int,
    b_addr: int,
    c_addr: int,
    k_dim: int = 1,
    id_tag: int = 0,
) -> tuple[int, int]:
    """Create COMPUTE command pair for execute controller."""
    flags = k_dim & 0xFFFF
    cmd0 = (ExecOpcode.COMPUTE << 56) | (id_tag << 48) | (flags << 32) | a_addr
    cmd1 = (b_addr << 32) | (c_addr & 0xFFFFFFFF)
    return (cmd0, cmd1)


# =============================================================================
# Legacy Functions (deprecated, use descriptor-based API instead)
# =============================================================================


def encode_command(
    opcode: int,
    data: int = 0,
    id_tag: int = 0,
    flags: int = 0,
) -> int:
    """
    Encode a 64-bit command word.

    DEPRECATED: Use descriptor-based API (DmaDescriptor, make_memcpy, etc.)
    for DMA operations. This function is retained for ExecuteController commands.
    """
    return (opcode << 56) | (id_tag << 48) | (flags << 32) | (data & 0xFFFFFFFF)


def decode_command(cmd: int) -> dict:
    """
    Decode a 64-bit command word.

    DEPRECATED: Use DmaDescriptor.from_bytes() for descriptor decoding.
    """
    opcode_val = (cmd >> 56) & 0xFF
    opcode: ExecOpcode | int
    try:
        opcode = ExecOpcode(opcode_val)
    except ValueError:
        opcode = opcode_val

    return {
        "opcode": opcode,
        "id_tag": (cmd >> 48) & 0xFF,
        "flags": (cmd >> 32) & 0xFFFF,
        "data": cmd & 0xFFFFFFFF,
    }
