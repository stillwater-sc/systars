"""
Command definitions for the systars accelerator.

This module defines the command opcodes and helper functions for encoding/decoding
commands sent to the systolic array controllers.

Command Format (simplified 64-bit):
    [63:56] opcode    - 8-bit operation code
    [55:48] id/tag    - 8-bit operation ID for tracking
    [47:32] flags     - 16-bit flags/configuration
    [31:0]  data      - 32-bit address or immediate data

For dual-address commands, a second 64-bit word provides the destination address.
"""

from enum import IntEnum


class OpCode(IntEnum):
    """
    Command opcodes for systars accelerator controllers.

    Organized by functional unit:
    - 0x00-0x0F: Configuration commands
    - 0x10-0x1F: Execute commands
    - 0x20-0x2F: Load commands
    - 0x30-0x3F: Store commands
    - 0xF0-0xFF: System commands
    """

    # Configuration commands
    CONFIG_EX = 0x00  # Configure execute controller
    CONFIG_LD = 0x01  # Configure load controller
    CONFIG_ST = 0x02  # Configure store controller

    # Execute commands
    PRELOAD = 0x10  # Preload bias/initial values from accumulator to array
    COMPUTE = 0x11  # Execute matmul: read from scratchpad, write to accumulator

    # Load commands (external memory -> scratchpad)
    MVIN = 0x20  # Move in: DRAM -> scratchpad
    MVIN2 = 0x21  # Move in with config set 1
    MVIN3 = 0x22  # Move in with config set 2

    # Store commands (accumulator -> external memory)
    MVOUT = 0x30  # Move out: accumulator -> DRAM

    # System commands
    FENCE = 0xF0  # Wait for all operations to complete
    FLUSH = 0xF1  # Flush all queues and reset state
    NOP = 0xFF  # No operation


class ConfigFlags(IntEnum):
    """
    Configuration flags for CONFIG commands.

    Used in the flags field [47:32] of CONFIG commands.
    """

    # Dataflow mode (2 bits)
    DATAFLOW_OS = 0x0000  # Output-stationary
    DATAFLOW_WS = 0x0001  # B-stationary (weight-stationary legacy name)
    DATAFLOW_IS = 0x0002  # A-stationary (input-stationary)

    # Accumulate mode
    ACCUMULATE = 0x0010  # Accumulate results (vs overwrite)

    # Activation function (3 bits)
    ACT_NONE = 0x0000
    ACT_RELU = 0x0100
    ACT_RELU6 = 0x0200


def encode_command(
    opcode: OpCode,
    data: int = 0,
    id_tag: int = 0,
    flags: int = 0,
) -> int:
    """
    Encode a 64-bit command word.

    Args:
        opcode: Operation code from OpCode enum
        data: 32-bit data field (address or immediate)
        id_tag: 8-bit operation ID for tracking (0-255)
        flags: 16-bit flags field

    Returns:
        64-bit encoded command
    """
    assert 0 <= opcode <= 0xFF, f"Invalid opcode: {opcode}"
    assert 0 <= id_tag <= 0xFF, f"Invalid id_tag: {id_tag}"
    assert 0 <= flags <= 0xFFFF, f"Invalid flags: {flags}"
    assert 0 <= data <= 0xFFFFFFFF, f"Invalid data: {data}"

    return (opcode << 56) | (id_tag << 48) | (flags << 32) | data


def decode_command(cmd: int) -> dict:
    """
    Decode a 64-bit command word.

    Args:
        cmd: 64-bit encoded command

    Returns:
        Dictionary with opcode, id_tag, flags, data fields
    """
    return {
        "opcode": OpCode((cmd >> 56) & 0xFF),
        "id_tag": (cmd >> 48) & 0xFF,
        "flags": (cmd >> 32) & 0xFFFF,
        "data": cmd & 0xFFFFFFFF,
    }


def make_config_ex(
    dataflow: int = 0,
    shift: int = 0,
    id_tag: int = 0,
) -> int:
    """
    Create CONFIG_EX command to configure execute controller.

    Args:
        dataflow: Dataflow mode (0=OS, 1=WS)
        shift: Rounding shift amount (0-31)
        id_tag: Operation ID

    Returns:
        64-bit encoded command
    """
    flags = dataflow & 0x0003  # Lower 2 bits for dataflow
    data = shift & 0x1F  # Lower 5 bits for shift
    return encode_command(OpCode.CONFIG_EX, data=data, id_tag=id_tag, flags=flags)


def make_preload(
    acc_addr: int,
    rows: int = 1,
    id_tag: int = 0,
) -> int:
    """
    Create PRELOAD command to load bias from accumulator.

    Args:
        acc_addr: Accumulator address (local address format)
        rows: Number of rows to preload
        id_tag: Operation ID

    Returns:
        64-bit encoded command
    """
    flags = rows & 0xFFFF
    return encode_command(OpCode.PRELOAD, data=acc_addr, id_tag=id_tag, flags=flags)


def make_compute(
    a_addr: int,
    b_addr: int,
    c_addr: int,
    k_dim: int = 1,
    id_tag: int = 0,
) -> tuple[int, int]:
    """
    Create COMPUTE command pair for matrix multiply.

    For simplicity, this returns two 64-bit words:
    - Word 0: opcode + A address
    - Word 1: B address (upper 32) + C address (lower 32)

    Args:
        a_addr: Scratchpad address for A matrix
        b_addr: Scratchpad address for B matrix
        c_addr: Accumulator address for C result
        k_dim: Inner dimension for the matmul
        id_tag: Operation ID

    Returns:
        Tuple of two 64-bit encoded commands
    """
    flags = k_dim & 0xFFFF
    cmd0 = encode_command(OpCode.COMPUTE, data=a_addr, id_tag=id_tag, flags=flags)
    cmd1 = (b_addr << 32) | (c_addr & 0xFFFFFFFF)
    return (cmd0, cmd1)


def make_mvin(
    dram_addr: int,
    sp_addr: int,
    len_bytes: int,
    id_tag: int = 0,
) -> tuple[int, int]:
    """
    Create MVIN command to load from DRAM to scratchpad.

    Args:
        dram_addr: Source address in external memory
        sp_addr: Destination address in scratchpad (local format)
        len_bytes: Transfer length in bytes
        id_tag: Operation ID

    Returns:
        Tuple of two 64-bit words (dram_addr, sp_addr | len)
    """
    flags = len_bytes & 0xFFFF
    cmd0 = encode_command(OpCode.MVIN, data=dram_addr & 0xFFFFFFFF, id_tag=id_tag, flags=flags)
    cmd1 = sp_addr & 0xFFFFFFFF
    return (cmd0, cmd1)


def make_mvout(
    acc_addr: int,
    dram_addr: int,
    len_bytes: int,
    activation: int = 0,
    id_tag: int = 0,
) -> tuple[int, int]:
    """
    Create MVOUT command to store from accumulator to DRAM.

    Args:
        acc_addr: Source address in accumulator (local format)
        dram_addr: Destination address in external memory
        len_bytes: Transfer length in bytes
        activation: Activation function to apply (0=none, 1=relu)
        id_tag: Operation ID

    Returns:
        Tuple of two 64-bit words
    """
    flags = (len_bytes & 0xFFF) | ((activation & 0xF) << 12)
    cmd0 = encode_command(OpCode.MVOUT, data=acc_addr, id_tag=id_tag, flags=flags)
    cmd1 = dram_addr & 0xFFFFFFFF
    return (cmd0, cmd1)
