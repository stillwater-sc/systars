"""
Local Address Encoding/Decoding Utilities.

This module provides utilities for encoding and decoding addresses in the
unified local memory address space (scratchpad + accumulator).

Address Layout (32 bits):
    [31]    is_acc        - 1 = accumulator, 0 = scratchpad
    [30]    accumulate    - 1 = accumulate mode (add to existing)
    [29]    read_full_row - 1 = read entire row width
    [28:0]  data          - Bank + row address

The lower bits of data contain:
    - Bank index (log2(num_banks) bits)
    - Row address within bank (remaining bits)
"""

from amaranth import Signal

from ..config import SystolicConfig


class LocalAddr:
    """
    Address encoding/decoding utilities for local memory.

    Provides methods to extract fields from a 32-bit local address,
    including flags, bank selection, and row addresses for both
    scratchpad and accumulator memory spaces.

    Example:
        >>> config = SystolicConfig(sp_banks=4, acc_banks=2)
        >>> addr_util = LocalAddr(config)
        >>> # In hardware:
        >>> bank = addr_util.sp_bank(address_signal)
        >>> row = addr_util.sp_row(address_signal)
    """

    # Bit positions for flag fields
    IS_ACC_BIT = 31
    ACCUMULATE_BIT = 30
    READ_FULL_ROW_BIT = 29
    DATA_BITS = 29  # Bits 0-28 contain the address data

    def __init__(self, config: SystolicConfig):
        """
        Initialize address utilities with configuration.

        Args:
            config: SystolicConfig with memory parameters
        """
        self.config = config

        # Compute bit widths for bank selection
        self.sp_bank_bits = max(1, (config.sp_banks - 1).bit_length())
        self.acc_bank_bits = max(1, (config.acc_banks - 1).bit_length())

        # Compute bit widths for row addresses
        self.sp_row_bits = max(1, (config.sp_bank_entries - 1).bit_length())
        self.acc_row_bits = max(1, (config.acc_bank_entries - 1).bit_length())

    @staticmethod
    def is_acc(addr: Signal) -> Signal:
        """
        Check if address refers to accumulator memory.

        Args:
            addr: 32-bit address signal

        Returns:
            Signal that is 1 if accumulator, 0 if scratchpad
        """
        return addr[LocalAddr.IS_ACC_BIT]

    @staticmethod
    def accumulate(addr: Signal) -> Signal:
        """
        Check if accumulate mode is set.

        When set, writes to accumulator should add to existing value
        rather than overwriting.

        Args:
            addr: 32-bit address signal

        Returns:
            Signal that is 1 if accumulate mode
        """
        return addr[LocalAddr.ACCUMULATE_BIT]

    @staticmethod
    def read_full_row(addr: Signal) -> Signal:
        """
        Check if full row read is requested.

        When set, read the entire row width instead of partial.

        Args:
            addr: 32-bit address signal

        Returns:
            Signal that is 1 if full row read
        """
        return addr[LocalAddr.READ_FULL_ROW_BIT]

    @staticmethod
    def data(addr: Signal) -> Signal:
        """
        Extract the raw address data (bank + row).

        Args:
            addr: 32-bit address signal

        Returns:
            Signal containing bits [28:0]
        """
        return addr[: LocalAddr.DATA_BITS]

    def sp_bank(self, addr: Signal) -> Signal:
        """
        Extract scratchpad bank index from address.

        The bank index is in the lowest bits of the data field.

        Args:
            addr: 32-bit address signal

        Returns:
            Signal containing bank index
        """
        return addr[: self.sp_bank_bits]

    def sp_row(self, addr: Signal) -> Signal:
        """
        Extract scratchpad row address within bank.

        The row address is above the bank bits in the data field.

        Args:
            addr: 32-bit address signal

        Returns:
            Signal containing row address within bank
        """
        return addr[self.sp_bank_bits : self.sp_bank_bits + self.sp_row_bits]

    def acc_bank(self, addr: Signal) -> Signal:
        """
        Extract accumulator bank index from address.

        Args:
            addr: 32-bit address signal

        Returns:
            Signal containing bank index
        """
        return addr[: self.acc_bank_bits]

    def acc_row(self, addr: Signal) -> Signal:
        """
        Extract accumulator row address within bank.

        Args:
            addr: 32-bit address signal

        Returns:
            Signal containing row address within bank
        """
        return addr[self.acc_bank_bits : self.acc_bank_bits + self.acc_row_bits]

    @staticmethod
    def is_garbage(addr: Signal) -> Signal:
        """
        Check if address is garbage (invalid).

        An address is considered garbage when all three flag bits are set.
        This pattern is used to indicate an invalid/null address.

        Args:
            addr: 32-bit address signal

        Returns:
            Signal that is 1 if address is garbage
        """
        return (
            addr[LocalAddr.IS_ACC_BIT]
            & addr[LocalAddr.ACCUMULATE_BIT]
            & addr[LocalAddr.READ_FULL_ROW_BIT]
        )

    @staticmethod
    def make_sp_addr(bank: int, row: int, read_full_row: bool = False) -> int:
        """
        Create a scratchpad address from components.

        Args:
            bank: Bank index
            row: Row address within bank
            read_full_row: Whether to read full row

        Returns:
            32-bit address integer
        """
        addr = bank | (row << max(1, (bank).bit_length() if bank > 0 else 1))
        if read_full_row:
            addr |= 1 << LocalAddr.READ_FULL_ROW_BIT
        return addr

    @staticmethod
    def make_acc_addr(
        bank: int, row: int, accumulate: bool = False, read_full_row: bool = False
    ) -> int:
        """
        Create an accumulator address from components.

        Args:
            bank: Bank index
            row: Row address within bank
            accumulate: Whether to use accumulate mode
            read_full_row: Whether to read full row

        Returns:
            32-bit address integer
        """
        addr = bank | (row << max(1, (bank).bit_length() if bank > 0 else 1))
        addr |= 1 << LocalAddr.IS_ACC_BIT  # Mark as accumulator
        if accumulate:
            addr |= 1 << LocalAddr.ACCUMULATE_BIT
        if read_full_row:
            addr |= 1 << LocalAddr.READ_FULL_ROW_BIT
        return addr

    @staticmethod
    def make_garbage_addr() -> int:
        """
        Create a garbage (invalid) address.

        Returns:
            32-bit garbage address with all flag bits set
        """
        return (
            (1 << LocalAddr.IS_ACC_BIT)
            | (1 << LocalAddr.ACCUMULATE_BIT)
            | (1 << LocalAddr.READ_FULL_ROW_BIT)
        )
