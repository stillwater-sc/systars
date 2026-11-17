"""
Local Address Encoding/Decoding Utilities.

This module provides utilities for encoding and decoding addresses in the
unified local memory address space (scratchpad + Combiner).

Address Layout (32 bits):
    [31]    is_combiner        - 1 = Combiner, 0 = scratchpad
    [30]    accumulate    - 1 = accumulate mode (add to existing)
    [29]    read_full_row - 1 = read entire row width
    [28:0]  data          - the row

A scratchpad (L2) address is a row of L2, linear: where the row lives (its bank
group and bank) is L2's business (``memory/scratchpad.py``, ``L2Layout``). A Combiner
address has its bank in the low bits of the data and the row within the bank above.

The encoding follows Gemmini's ``local_addr_t`` (ucb-bar/gemmini; see NOTICE).
"""

from amaranth import Signal

from ..config import SystolicConfig


class LocalAddr:
    """
    Address encoding/decoding utilities for local memory.

    Provides methods to extract fields from a 32-bit local address,
    including flags, and the Combiner's bank selection and row addresses.

    Example:
        >>> config = SystolicConfig(combiner_banks=2)
        >>> addr_util = LocalAddr(config)
        >>> # In hardware:
        >>> bank = addr_util.combiner_bank(address_signal)
        >>> row = addr_util.combiner_row(address_signal)
    """

    # Bit positions for flag fields
    IS_COMBINER_BIT = 31
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

        # Bank selection bits: log2(banks), so one bank takes no address bits (with a
        # bit reserved, every other row would select a bank that does not exist)
        self.combiner_bank_bits = (config.combiner_banks - 1).bit_length()

        # Compute bit widths for row addresses
        self.combiner_row_bits = max(1, (config.combiner_bank_entries - 1).bit_length())

    @staticmethod
    def is_combiner(addr: Signal) -> Signal:
        """
        Check if address refers to Combiner memory.

        Args:
            addr: 32-bit address signal

        Returns:
            Signal that is 1 if Combiner, 0 if scratchpad
        """
        return addr[LocalAddr.IS_COMBINER_BIT]

    @staticmethod
    def accumulate(addr: Signal) -> Signal:
        """
        Check if accumulate mode is set.

        When set, writes to Combiner should add to existing value
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

    def combiner_bank(self, addr: Signal) -> Signal:
        """
        Extract Combiner bank index from address.

        Args:
            addr: 32-bit address signal

        Returns:
            Signal containing bank index
        """
        return addr[: self.combiner_bank_bits]

    def combiner_row(self, addr: Signal) -> Signal:
        """
        Extract Combiner row address within bank.

        Args:
            addr: 32-bit address signal

        Returns:
            Signal containing row address within bank
        """
        return addr[self.combiner_bank_bits : self.combiner_bank_bits + self.combiner_row_bits]

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
            addr[LocalAddr.IS_COMBINER_BIT]
            & addr[LocalAddr.ACCUMULATE_BIT]
            & addr[LocalAddr.READ_FULL_ROW_BIT]
        )

    def make_combiner_addr(
        self, bank: int, row: int, accumulate: bool = False, read_full_row: bool = False
    ) -> int:
        """
        Create a Combiner address from components.

        Args:
            bank: Bank index
            row: Row address within bank
            accumulate: Whether to use accumulate mode
            read_full_row: Whether to read full row

        Returns:
            32-bit address integer
        """
        # Bank in the low combiner_bank_bits, row above it (matches the hardware decode)
        addr = bank | (row << self.combiner_bank_bits)
        addr |= 1 << LocalAddr.IS_COMBINER_BIT  # Mark as Combiner
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
            (1 << LocalAddr.IS_COMBINER_BIT)
            | (1 << LocalAddr.ACCUMULATE_BIT)
            | (1 << LocalAddr.READ_FULL_ROW_BIT)
        )
