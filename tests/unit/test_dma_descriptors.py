"""
Unit tests for the DMA descriptor API.

These tests verify:
1. Descriptor creation and serialization
2. Descriptor chain building
3. Memory region utilities
4. Flag handling
"""

import struct

import pytest

from systars.util.commands import (
    DescriptorChain,
    DmaDescriptor,
    DmaFlags,
    DmaOpcode,
    MemoryRegion,
    get_memory_region,
    is_local_memory,
    make_crc32,
    make_fence,
    make_fill,
    make_memcpy,
)


class TestDmaOpcode:
    """Test DMA opcode definitions."""

    def test_memcpy_opcode(self):
        """Test MEMCPY opcode value."""
        assert DmaOpcode.MEMCPY == 0x00

    def test_fill_opcode(self):
        """Test FILL opcode value."""
        assert DmaOpcode.FILL == 0x01

    def test_crc32_opcode(self):
        """Test CRC32 opcode value."""
        assert DmaOpcode.CRC32 == 0x10

    def test_fence_opcode(self):
        """Test FENCE opcode value."""
        assert DmaOpcode.FENCE == 0x20


class TestDmaFlags:
    """Test DMA flags."""

    def test_chain_flag(self):
        """Test CHAIN flag."""
        assert DmaFlags.CHAIN == 0x01

    def test_interrupt_flag(self):
        """Test INTERRUPT_ON_COMPLETE flag."""
        assert DmaFlags.INTERRUPT_ON_COMPLETE == 0x02

    def test_flag_combination(self):
        """Test combining flags."""
        flags = DmaFlags.CHAIN | DmaFlags.INTERRUPT_ON_COMPLETE
        assert flags & DmaFlags.CHAIN
        assert flags & DmaFlags.INTERRUPT_ON_COMPLETE
        assert not (flags & DmaFlags.SRC_IS_PATTERN)


class TestMemoryRegion:
    """Test memory region utilities."""

    def test_dram_region(self):
        """Test DRAM region detection."""
        assert get_memory_region(0x0) == "DRAM"
        assert get_memory_region(0x1000) == "DRAM"
        assert get_memory_region(0xFFFF_FFFF) == "DRAM"

    def test_scratchpad_region(self):
        """Test scratchpad region detection."""
        assert get_memory_region(MemoryRegion.SCRATCHPAD_BASE) == "SCRATCHPAD"
        assert get_memory_region(MemoryRegion.SCRATCHPAD_BASE + 0x100) == "SCRATCHPAD"

    def test_accumulator_region(self):
        """Test accumulator region detection."""
        assert get_memory_region(MemoryRegion.ACCUMULATOR_BASE) == "ACCUMULATOR"
        assert get_memory_region(MemoryRegion.ACCUMULATOR_BASE + 0x100) == "ACCUMULATOR"

    def test_is_local_memory(self):
        """Test local memory detection."""
        assert not is_local_memory(0x0)
        assert not is_local_memory(0x1000)
        assert is_local_memory(MemoryRegion.SCRATCHPAD_BASE)
        assert is_local_memory(MemoryRegion.ACCUMULATOR_BASE)


class TestDmaDescriptor:
    """Test DmaDescriptor dataclass."""

    def test_basic_creation(self):
        """Test basic descriptor creation."""
        desc = DmaDescriptor(
            opcode=DmaOpcode.MEMCPY,
            length=4096,
            src_addr=0x1000,
            dst_addr=0x2000,
        )
        assert desc.opcode == DmaOpcode.MEMCPY
        assert desc.length == 4096
        assert desc.src_addr == 0x1000
        assert desc.dst_addr == 0x2000
        assert desc.next_desc == 0
        assert desc.flags == DmaFlags.NONE

    def test_to_bytes_size(self):
        """Test that serialized descriptor is 64 bytes."""
        desc = DmaDescriptor(opcode=DmaOpcode.MEMCPY)
        data = desc.to_bytes()
        assert len(data) == 64

    def test_to_bytes_content(self):
        """Test descriptor serialization content."""
        desc = DmaDescriptor(
            opcode=DmaOpcode.MEMCPY,
            flags=DmaFlags.CHAIN,
            length=0x1234,
            src_addr=0xAAAA_BBBB_CCCC_DDDD,
            dst_addr=0x1111_2222_3333_4444,
        )
        data = desc.to_bytes()

        # Parse first few fields manually
        opcode = data[0]
        flags = data[1]
        length = struct.unpack("<I", data[4:8])[0]
        src_addr = struct.unpack("<Q", data[8:16])[0]
        dst_addr = struct.unpack("<Q", data[16:24])[0]

        assert opcode == DmaOpcode.MEMCPY
        assert flags == DmaFlags.CHAIN
        assert length == 0x1234
        assert src_addr == 0xAAAA_BBBB_CCCC_DDDD
        assert dst_addr == 0x1111_2222_3333_4444

    def test_from_bytes_roundtrip(self):
        """Test serialization/deserialization roundtrip."""
        original = DmaDescriptor(
            opcode=DmaOpcode.FILL,
            flags=DmaFlags.CHAIN | DmaFlags.INTERRUPT_ON_COMPLETE,
            length=8192,
            src_addr=0xDEADBEEF,  # Pattern for FILL
            dst_addr=MemoryRegion.SCRATCHPAD_BASE,
            next_desc=0x1000,
            completion=0x2000,
            user_data=0x42,
        )
        data = original.to_bytes()
        restored = DmaDescriptor.from_bytes(data)

        assert restored.opcode == original.opcode
        assert restored.flags == original.flags
        assert restored.length == original.length
        assert restored.src_addr == original.src_addr
        assert restored.dst_addr == original.dst_addr
        assert restored.next_desc == original.next_desc
        assert restored.completion == original.completion
        assert restored.user_data == original.user_data

    def test_repr(self):
        """Test descriptor string representation."""
        desc = DmaDescriptor(
            opcode=DmaOpcode.MEMCPY,
            src_addr=0x1000,
            dst_addr=0x2000,
            length=4096,
        )
        repr_str = repr(desc)
        assert "MEMCPY" in repr_str
        assert "0x001000" in repr_str or "1000" in repr_str


class TestMakeMemcpy:
    """Test make_memcpy helper function."""

    def test_basic_memcpy(self):
        """Test basic MEMCPY descriptor creation."""
        desc = make_memcpy(
            src_addr=0x1000,
            dst_addr=0x2000,
            length=4096,
        )
        assert desc.opcode == DmaOpcode.MEMCPY
        assert desc.src_addr == 0x1000
        assert desc.dst_addr == 0x2000
        assert desc.length == 4096
        assert desc.next_desc == 0
        assert not (desc.flags & DmaFlags.CHAIN)

    def test_memcpy_with_chain(self):
        """Test MEMCPY with chain pointer."""
        desc = make_memcpy(
            src_addr=0x1000,
            dst_addr=0x2000,
            length=4096,
            next_desc=0x3000,
        )
        assert desc.next_desc == 0x3000
        assert desc.flags & DmaFlags.CHAIN

    def test_memcpy_with_interrupt(self):
        """Test MEMCPY with interrupt on complete."""
        desc = make_memcpy(
            src_addr=0x1000,
            dst_addr=0x2000,
            length=4096,
            interrupt=True,
        )
        assert desc.flags & DmaFlags.INTERRUPT_ON_COMPLETE

    def test_memcpy_local_to_dram(self):
        """Test MEMCPY from local memory to DRAM."""
        desc = make_memcpy(
            src_addr=MemoryRegion.SCRATCHPAD_BASE,
            dst_addr=0x1000,
            length=4096,
        )
        assert desc.flags & DmaFlags.SRC_IS_LOCAL
        assert not (desc.flags & DmaFlags.DST_IS_LOCAL)

    def test_memcpy_dram_to_local(self):
        """Test MEMCPY from DRAM to local memory."""
        desc = make_memcpy(
            src_addr=0x1000,
            dst_addr=MemoryRegion.ACCUMULATOR_BASE,
            length=4096,
        )
        assert not (desc.flags & DmaFlags.SRC_IS_LOCAL)
        assert desc.flags & DmaFlags.DST_IS_LOCAL


class TestMakeFill:
    """Test make_fill helper function."""

    def test_basic_fill(self):
        """Test basic FILL descriptor creation."""
        desc = make_fill(
            dst_addr=MemoryRegion.SCRATCHPAD_BASE,
            length=4096,
            pattern=0,
        )
        assert desc.opcode == DmaOpcode.FILL
        assert desc.dst_addr == MemoryRegion.SCRATCHPAD_BASE
        assert desc.length == 4096
        assert desc.src_addr == 0  # Pattern
        assert desc.flags & DmaFlags.SRC_IS_PATTERN

    def test_fill_with_pattern(self):
        """Test FILL with non-zero pattern."""
        desc = make_fill(
            dst_addr=MemoryRegion.ACCUMULATOR_BASE,
            length=1024,
            pattern=0xDEADBEEF_CAFEBABE,
        )
        assert desc.src_addr == 0xDEADBEEF_CAFEBABE


class TestMakeFence:
    """Test make_fence helper function."""

    def test_basic_fence(self):
        """Test basic FENCE descriptor creation."""
        desc = make_fence()
        assert desc.opcode == DmaOpcode.FENCE
        assert desc.length == 0
        assert desc.next_desc == 0

    def test_fence_with_chain(self):
        """Test FENCE with chain pointer."""
        desc = make_fence(next_desc=0x1000)
        assert desc.next_desc == 0x1000
        assert desc.flags & DmaFlags.CHAIN


class TestMakeCrc32:
    """Test make_crc32 helper function."""

    def test_basic_crc32(self):
        """Test basic CRC32 descriptor creation."""
        desc = make_crc32(
            src_addr=0x1000,
            length=4096,
            result_addr=0x2000,
        )
        assert desc.opcode == DmaOpcode.CRC32
        assert desc.src_addr == 0x1000
        assert desc.length == 4096
        assert desc.dst_addr == 0x2000


class TestDescriptorChain:
    """Test DescriptorChain builder."""

    def test_empty_chain(self):
        """Test building empty chain."""
        chain = DescriptorChain()
        result = chain.build()
        assert result == []

    def test_single_memcpy(self):
        """Test chain with single MEMCPY."""
        chain = DescriptorChain(base_addr=0x1000)
        chain.memcpy(src=0x0, dst=MemoryRegion.SCRATCHPAD_BASE, length=4096)
        result = chain.build()

        assert len(result) == 1
        assert result[0].opcode == DmaOpcode.MEMCPY
        assert result[0].next_desc == 0  # End of chain
        assert not (result[0].flags & DmaFlags.CHAIN)

    def test_multiple_operations(self):
        """Test chain with multiple operations."""
        chain = DescriptorChain(base_addr=0x1000)
        chain.memcpy(src=0x0, dst=MemoryRegion.SCRATCHPAD_BASE, length=4096)
        chain.memcpy(src=0x1000, dst=MemoryRegion.SCRATCHPAD_BASE + 4096, length=4096)
        chain.fence()
        result = chain.build()

        assert len(result) == 3

        # First descriptor chains to second
        assert result[0].next_desc == 0x1000 + 64
        assert result[0].flags & DmaFlags.CHAIN

        # Second chains to third
        assert result[1].next_desc == 0x1000 + 128
        assert result[1].flags & DmaFlags.CHAIN

        # Third is end of chain
        assert result[2].next_desc == 0
        assert not (result[2].flags & DmaFlags.CHAIN)

    def test_fluent_interface(self):
        """Test fluent builder interface."""
        chain = (
            DescriptorChain(base_addr=0x2000)
            .fill(dst=MemoryRegion.ACCUMULATOR_BASE, length=1024, pattern=0)
            .memcpy(src=0x0, dst=MemoryRegion.SCRATCHPAD_BASE, length=4096)
            .fence(interrupt=True)
        )
        result = chain.build()

        assert len(result) == 3
        assert result[0].opcode == DmaOpcode.FILL
        assert result[1].opcode == DmaOpcode.MEMCPY
        assert result[2].opcode == DmaOpcode.FENCE
        assert result[2].flags & DmaFlags.INTERRUPT_ON_COMPLETE

    def test_to_bytes(self):
        """Test converting chain to bytes."""
        chain = DescriptorChain(base_addr=0x1000)
        chain.memcpy(src=0x0, dst=MemoryRegion.SCRATCHPAD_BASE, length=4096)
        chain.fence()
        chain.build()

        data = chain.to_bytes()
        assert len(data) == 128  # 2 descriptors * 64 bytes

    def test_completion_address(self):
        """Test setting completion address on final descriptor."""
        chain = DescriptorChain(base_addr=0x1000)
        chain.memcpy(src=0x0, dst=MemoryRegion.SCRATCHPAD_BASE, length=4096)
        result = chain.build(completion_addr=0x3000)

        assert result[0].completion == 0x3000
        assert result[0].flags & DmaFlags.WRITEBACK_STATUS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
