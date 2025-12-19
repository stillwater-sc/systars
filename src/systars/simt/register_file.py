"""
Banked Register File for SIMT Streaming Multiprocessor.

The register file is organized into banks to enable parallel access.
Bank conflicts occur when multiple operands map to the same bank,
causing pipeline stalls and additional energy consumption.

Architecture (per partition):
    ┌──────────────────────────────────────────────────────────────────┐
    │                    REGISTER FILE (16K Registers)                  │
    │                                                                   │
    │  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐  │
    │  │ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │ B8 │ B9 │... │B15│  │
    │  │1K  │1K  │1K  │1K  │1K  │1K  │1K  │1K  │1K  │1K  │    │1K │  │
    │  └──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴────┴──┬─┘  │
    │     │    │    │    │    │    │    │    │    │    │         │     │
    │     └────┴────┴────┴────┴────┴────┴────┴────┴────┴─────────┘     │
    │                          │                                        │
    │                ┌─────────┴─────────┐                             │
    │                │  Bank Arbiter     │                             │
    │                │  (Conflict Det.)  │                             │
    │                └─────────┬─────────┘                             │
    │                          │                                        │
    │             ┌────────────┼────────────┐                          │
    │             ▼            ▼            ▼                          │
    │         Read Port 0  Read Port 1  Read Port 2                    │
    │         (src1)       (src2)       (src3)                         │
    └──────────────────────────────────────────────────────────────────┘

Bank Selection:
    bank_id = register_address % num_banks

Bank Conflict:
    Occurs when multiple reads target the same bank in one cycle.
    Resolution: serialize accesses, causing 1-cycle stall per conflict.
"""

from dataclasses import dataclass, field
from enum import IntEnum, auto

from amaranth import Module, Signal, unsigned
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import Component, In, Out

from .config import SIMTConfig


class BankState(IntEnum):
    """Bank access state for animation."""

    IDLE = 0
    READ = auto()
    WRITE = auto()
    CONFLICT = auto()


@dataclass
class RegisterAccess:
    """Represents a register file access request."""

    register_id: int
    is_write: bool = False
    data: int = 0
    thread_id: int = 0

    @property
    def bank_id(self) -> int:
        """Compute which bank this register maps to."""
        return self.register_id % 16  # 16 banks


@dataclass
class BankStatus:
    """Status of a single register bank."""

    bank_id: int
    state: BankState = BankState.IDLE
    accessing_threads: list = field(default_factory=list)
    conflict_count: int = 0


class RegisterFileBank(Component):
    """
    Single register file bank (SRAM).

    Each bank stores registers_per_bank registers.
    Has single read/write port - conflicts resolved by arbiter.

    Ports:
        addr: Register address within bank
        read_en: Enable read operation
        write_en: Enable write operation
        write_data: Data to write
        read_data: Data read (1-cycle latency)
        read_valid: Read data valid
    """

    def __init__(self, config: SIMTConfig, bank_id: int = 0):
        """
        Initialize a register file bank.

        Args:
            config: SIMT configuration
            bank_id: Identifier for this bank
        """
        self.config = config
        self.bank_id = bank_id

        # Address width for registers_per_bank entries
        self.addr_bits = max(1, config.registers_per_bank.bit_length())

        super().__init__(
            {
                # Address and control
                "addr": In(unsigned(self.addr_bits)),
                "read_en": In(1),
                "write_en": In(1),
                "write_data": In(unsigned(config.register_width)),
                # Output
                "read_data": Out(unsigned(config.register_width)),
                "read_valid": Out(1),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        # Create memory for this bank
        mem = Memory(
            shape=unsigned(cfg.register_width),
            depth=cfg.registers_per_bank,
            init=[],
        )
        m.submodules.mem = mem

        # Read port with 1-cycle latency
        rd_port = mem.read_port()
        m.d.comb += [
            rd_port.addr.eq(self.addr),
            rd_port.en.eq(self.read_en),
            self.read_data.eq(rd_port.data),
        ]

        # Track read valid (1-cycle latency)
        m.d.sync += self.read_valid.eq(self.read_en)

        # Write port
        wr_port = mem.write_port()
        m.d.comb += [
            wr_port.addr.eq(self.addr),
            wr_port.data.eq(self.write_data),
            wr_port.en.eq(self.write_en),
        ]

        return m


class PartitionRegisterFile(Component):
    """
    Complete register file for one partition with bank conflict detection.

    Contains 16 banks with conflict detection and arbitration.
    Supports 3 read ports (for 3 source operands) and 1 write port.

    Ports:
        # Read port 0 (src1)
        read0_addr: Register address for read port 0
        read0_en: Enable read port 0
        read0_data: Data from read port 0
        read0_valid: Read port 0 data valid

        # Read port 1 (src2)
        read1_addr: Register address for read port 1
        read1_en: Enable read port 1
        read1_data: Data from read port 1
        read1_valid: Read port 1 data valid

        # Read port 2 (src3)
        read2_addr: Register address for read port 2
        read2_en: Enable read port 2
        read2_data: Data from read port 2
        read2_valid: Read port 2 data valid

        # Write port
        write_addr: Register address for write
        write_en: Enable write
        write_data: Data to write

        # Status
        conflict: Bank conflict detected this cycle
        conflict_bank: Which bank has conflict
    """

    def __init__(self, config: SIMTConfig, partition_id: int = 0):
        """
        Initialize partition register file.

        Args:
            config: SIMT configuration
            partition_id: Identifier for this partition
        """
        self.config = config
        self.partition_id = partition_id

        # Address bits for full register space
        self.addr_bits = max(1, config.registers_per_partition.bit_length())
        self.bank_bits = max(1, (config.register_banks_per_partition - 1).bit_length())

        super().__init__(
            {
                # Read port 0
                "read0_addr": In(unsigned(self.addr_bits)),
                "read0_en": In(1),
                "read0_data": Out(unsigned(config.register_width)),
                "read0_valid": Out(1),
                # Read port 1
                "read1_addr": In(unsigned(self.addr_bits)),
                "read1_en": In(1),
                "read1_data": Out(unsigned(config.register_width)),
                "read1_valid": Out(1),
                # Read port 2
                "read2_addr": In(unsigned(self.addr_bits)),
                "read2_en": In(1),
                "read2_data": Out(unsigned(config.register_width)),
                "read2_valid": Out(1),
                # Write port
                "write_addr": In(unsigned(self.addr_bits)),
                "write_en": In(1),
                "write_data": In(unsigned(config.register_width)),
                # Status
                "conflict": Out(1),
                "conflict_bank": Out(unsigned(self.bank_bits)),
                "stall": Out(1),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config
        num_banks = cfg.register_banks_per_partition

        # Instantiate all banks
        banks = [RegisterFileBank(cfg, i) for i in range(num_banks)]
        for i, bank in enumerate(banks):
            m.submodules[f"bank_{i}"] = bank

        # Compute bank indices for each port
        # bank_id = addr % num_banks = addr[0:bank_bits]
        read0_bank = Signal(unsigned(self.bank_bits), name="read0_bank")
        read1_bank = Signal(unsigned(self.bank_bits), name="read1_bank")
        read2_bank = Signal(unsigned(self.bank_bits), name="read2_bank")
        write_bank = Signal(unsigned(self.bank_bits), name="write_bank")

        m.d.comb += [
            read0_bank.eq(self.read0_addr[: self.bank_bits]),
            read1_bank.eq(self.read1_addr[: self.bank_bits]),
            read2_bank.eq(self.read2_addr[: self.bank_bits]),
            write_bank.eq(self.write_addr[: self.bank_bits]),
        ]

        # Compute in-bank addresses (addr // num_banks)
        bank_addr_bits = max(1, cfg.registers_per_bank.bit_length())
        read0_bank_addr = Signal(unsigned(bank_addr_bits), name="read0_bank_addr")
        read1_bank_addr = Signal(unsigned(bank_addr_bits), name="read1_bank_addr")
        read2_bank_addr = Signal(unsigned(bank_addr_bits), name="read2_bank_addr")
        write_bank_addr = Signal(unsigned(bank_addr_bits), name="write_bank_addr")

        m.d.comb += [
            read0_bank_addr.eq(self.read0_addr >> self.bank_bits),
            read1_bank_addr.eq(self.read1_addr >> self.bank_bits),
            read2_bank_addr.eq(self.read2_addr >> self.bank_bits),
            write_bank_addr.eq(self.write_addr >> self.bank_bits),
        ]

        # Bank conflict detection
        # Conflict if any two read ports target the same bank
        conflict_01 = Signal(name="conflict_01")
        conflict_02 = Signal(name="conflict_02")
        conflict_12 = Signal(name="conflict_12")

        m.d.comb += [
            conflict_01.eq(self.read0_en & self.read1_en & (read0_bank == read1_bank)),
            conflict_02.eq(self.read0_en & self.read2_en & (read0_bank == read2_bank)),
            conflict_12.eq(self.read1_en & self.read2_en & (read1_bank == read2_bank)),
        ]

        any_conflict = Signal(name="any_conflict")
        m.d.comb += any_conflict.eq(conflict_01 | conflict_02 | conflict_12)
        m.d.comb += self.conflict.eq(any_conflict)
        m.d.comb += self.stall.eq(any_conflict)

        # Report first conflicting bank
        with m.If(conflict_01):
            m.d.comb += self.conflict_bank.eq(read0_bank)
        with m.Elif(conflict_02):
            m.d.comb += self.conflict_bank.eq(read0_bank)
        with m.Elif(conflict_12):
            m.d.comb += self.conflict_bank.eq(read1_bank)

        # Route read requests to banks
        # For simplicity, in case of conflict we stall (no replay logic here)
        for i, bank in enumerate(banks):
            # Read from this bank if any port targets it
            read0_hit = Signal(name=f"read0_hit_{i}")
            read1_hit = Signal(name=f"read1_hit_{i}")
            read2_hit = Signal(name=f"read2_hit_{i}")
            write_hit = Signal(name=f"write_hit_{i}")

            m.d.comb += [
                read0_hit.eq(self.read0_en & (read0_bank == i)),
                read1_hit.eq(self.read1_en & (read1_bank == i)),
                read2_hit.eq(self.read2_en & (read2_bank == i)),
                write_hit.eq(self.write_en & (write_bank == i)),
            ]

            # Priority: write > read0 > read1 > read2
            # In conflict case, only first reader wins
            with m.If(write_hit):
                m.d.comb += [
                    bank.addr.eq(write_bank_addr),
                    bank.write_en.eq(1),
                    bank.write_data.eq(self.write_data),
                    bank.read_en.eq(0),
                ]
            with m.Elif(read0_hit):
                m.d.comb += [
                    bank.addr.eq(read0_bank_addr),
                    bank.read_en.eq(1),
                    bank.write_en.eq(0),
                ]
            with m.Elif(read1_hit & ~conflict_01):
                m.d.comb += [
                    bank.addr.eq(read1_bank_addr),
                    bank.read_en.eq(1),
                    bank.write_en.eq(0),
                ]
            with m.Elif(read2_hit & ~conflict_02 & ~conflict_12):
                m.d.comb += [
                    bank.addr.eq(read2_bank_addr),
                    bank.read_en.eq(1),
                    bank.write_en.eq(0),
                ]
            with m.Else():
                m.d.comb += [
                    bank.read_en.eq(0),
                    bank.write_en.eq(0),
                ]

        # Collect read data from banks
        # Each output port gets data from its target bank
        for i, bank in enumerate(banks):
            with m.If(read0_bank == i):
                m.d.comb += self.read0_data.eq(bank.read_data)
                m.d.comb += self.read0_valid.eq(bank.read_valid)

            with m.If(read1_bank == i):
                m.d.comb += self.read1_data.eq(bank.read_data)
                m.d.comb += self.read1_valid.eq(bank.read_valid & ~conflict_01)

            with m.If(read2_bank == i):
                m.d.comb += self.read2_data.eq(bank.read_data)
                m.d.comb += self.read2_valid.eq(bank.read_valid & ~conflict_02 & ~conflict_12)

        return m


# =============================================================================
# Simulation Model for Animation
# =============================================================================


@dataclass
class RegisterFileSim:
    """
    Behavioral simulation model of partition register file.

    Used for animation and energy estimation without RTL simulation.
    """

    config: SIMTConfig
    partition_id: int = 0

    # Storage: list of register values
    registers: list = field(default_factory=list)

    # Bank status for visualization
    bank_status: list = field(default_factory=list)

    # Statistics
    total_reads: int = 0
    total_writes: int = 0
    total_conflicts: int = 0
    total_energy_pj: float = 0.0

    def __post_init__(self):
        """Initialize register storage and bank status."""
        self.registers = [0] * self.config.registers_per_partition
        self.bank_status = [
            BankStatus(bank_id=i) for i in range(self.config.register_banks_per_partition)
        ]

    def _get_bank(self, reg_addr: int) -> int:
        """Get bank ID for a register address."""
        return reg_addr % self.config.register_banks_per_partition

    def _get_bank_addr(self, reg_addr: int) -> int:
        """Get address within bank."""
        return reg_addr // self.config.register_banks_per_partition

    def read(self, reg_addrs: list[int]) -> tuple[list[int], bool, list[int]]:
        """
        Read multiple registers, detecting bank conflicts.

        Args:
            reg_addrs: List of register addresses to read

        Returns:
            Tuple of (data_values, has_conflict, conflicting_banks)
        """
        # Reset bank status
        for bank in self.bank_status:
            bank.state = BankState.IDLE
            bank.accessing_threads = []
            bank.conflict_count = 0

        # Track which banks are being accessed
        bank_accesses: dict[int, list[int]] = {}

        for i, addr in enumerate(reg_addrs):
            if addr is not None:
                bank_id = self._get_bank(addr)
                if bank_id not in bank_accesses:
                    bank_accesses[bank_id] = []
                bank_accesses[bank_id].append(i)

        # Detect conflicts
        conflicting_banks = []
        for bank_id, accessors in bank_accesses.items():
            if len(accessors) > 1:
                conflicting_banks.append(bank_id)
                self.bank_status[bank_id].state = BankState.CONFLICT
                self.bank_status[bank_id].conflict_count = len(accessors) - 1
                self.total_conflicts += len(accessors) - 1
            elif len(accessors) == 1:
                self.bank_status[bank_id].state = BankState.READ
            self.bank_status[bank_id].accessing_threads = accessors

        # Read data (in simulation, we can read regardless of conflict)
        data = []
        for addr in reg_addrs:
            if addr is not None:
                data.append(self.registers[addr])
                self.total_reads += 1
                self.total_energy_pj += self.config.register_read_energy_pj
            else:
                data.append(0)

        # Add conflict energy
        for _bank_id in conflicting_banks:
            self.total_energy_pj += self.config.bank_conflict_energy_pj

        has_conflict = len(conflicting_banks) > 0
        return data, has_conflict, conflicting_banks

    def write(self, reg_addr: int, value: int) -> None:
        """
        Write a register.

        Args:
            reg_addr: Register address
            value: Value to write
        """
        bank_id = self._get_bank(reg_addr)
        self.registers[reg_addr] = value
        self.bank_status[bank_id].state = BankState.WRITE
        self.total_writes += 1
        self.total_energy_pj += self.config.register_write_energy_pj

    def get_bank_visualization(self) -> list[str]:
        """Get ASCII visualization of bank states."""
        symbols = {
            BankState.IDLE: "░░",
            BankState.READ: "██",
            BankState.WRITE: "▓▓",
            BankState.CONFLICT: "XX",
        }
        return [symbols[bank.state] for bank in self.bank_status]

    def reset_cycle(self) -> None:
        """Reset bank status for new cycle."""
        for bank in self.bank_status:
            bank.state = BankState.IDLE
            bank.accessing_threads = []
