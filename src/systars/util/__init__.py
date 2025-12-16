"""Utility modules for systars accelerator."""

from .commands import (
    ConfigFlags,
    DescriptorChain,
    # Descriptor-based DMA API
    DmaDescriptor,
    DmaFlags,
    DmaOpcode,
    # Execute controller API
    ExecOpcode,
    MemoryRegion,
    # Legacy execute command encoding (for backward compatibility)
    OpCode,  # Alias for ExecOpcode
    decode_command,
    encode_command,
    get_memory_region,
    is_local_memory,
    make_compute,
    make_config_ex,
    make_crc32,
    make_fence,
    make_fill,
    make_memcpy,
    make_preload,
)

__all__ = [
    # Descriptor-based DMA API
    "DmaDescriptor",
    "DmaFlags",
    "DmaOpcode",
    "DescriptorChain",
    "MemoryRegion",
    "make_memcpy",
    "make_fill",
    "make_fence",
    "make_crc32",
    "get_memory_region",
    "is_local_memory",
    # Execute controller API
    "ExecOpcode",
    "ConfigFlags",
    "make_config_ex",
    "make_preload",
    "make_compute",
    # Legacy execute command encoding
    "OpCode",
    "encode_command",
    "decode_command",
]
