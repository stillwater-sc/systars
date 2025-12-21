"""
MasPar Processing Element (PE).

Each PE in the MasPar array contains:
- 64 x 32-bit registers (MP-2) or 48 registers (MP-1)
- 64KB local SRAM
- 32-bit ALU (MP-2) or 4-bit ALU (MP-1)
- Optional FPU
- Predicate/mask bit for conditional execution
"""

from dataclasses import dataclass, field

import numpy as np

from .config import MasParConfig
from .instruction import Instruction, Opcode


@dataclass
class PE:
    """
    MasPar Processing Element.

    PEs execute instructions broadcast from the ACU. Each PE operates on its
    own local data (registers and memory) in SIMD fashion.

    Attributes:
        row: Row position in the PE array
        col: Column position in the PE array
        config: Hardware configuration
        registers: Register file (64 x 32-bit by default)
        local_memory: Local SRAM (64KB by default)
        active: Whether PE participates in current instruction
        predicate: Result of last comparison (for conditional masking)
    """

    row: int
    col: int
    config: MasParConfig

    # Register file - initialized after __init__
    registers: np.ndarray = field(init=False)

    # Local memory - initialized after __init__
    local_memory: np.ndarray = field(init=False)

    # PE state
    active: bool = True  # Controlled by SETMASK/CLRMASK
    predicate: bool = False  # Set by comparison instructions

    # ALU state for multi-cycle operations
    alu_busy_cycles: int = 0
    pending_result: int = 0
    pending_dst: int = 0

    def __post_init__(self) -> None:
        """Initialize register file and local memory."""
        # Register file: 64 x 32-bit integers (use int32 for proper overflow)
        self.registers = np.zeros(self.config.registers_per_pe, dtype=np.int32)

        # Local memory: 64KB as byte array
        self.local_memory = np.zeros(self.config.local_memory_bytes, dtype=np.uint8)

    def reset(self) -> None:
        """Reset PE state."""
        self.registers.fill(0)
        self.local_memory.fill(0)
        self.active = True
        self.predicate = False
        self.alu_busy_cycles = 0
        self.pending_result = 0
        self.pending_dst = 0

    def execute(self, instr: Instruction) -> int | None:
        """
        Execute an instruction.

        Args:
            instr: Instruction to execute

        Returns:
            Result value for operations that produce one, None otherwise.
            For multi-cycle ops, returns None until completion.
        """
        if not self.active:
            return None

        op = instr.opcode
        dst = instr.dst
        src1_val = self.registers[instr.src1]
        src2_val = self.registers[instr.src2]

        # Convert to Python int for operations to avoid overflow issues
        src1 = int(src1_val)
        src2 = int(src2_val)

        result: int | None = None

        # Integer ALU operations
        if op == Opcode.NOP:
            pass
        elif op == Opcode.IADD:
            result = (src1 + src2) & 0xFFFFFFFF
            self.registers[dst] = np.int32(result if result < 0x80000000 else result - 0x100000000)
        elif op == Opcode.ISUB:
            result = (src1 - src2) & 0xFFFFFFFF
            self.registers[dst] = np.int32(result if result < 0x80000000 else result - 0x100000000)
        elif op == Opcode.IMUL:
            result = (src1 * src2) & 0xFFFFFFFF
            self.registers[dst] = np.int32(result if result < 0x80000000 else result - 0x100000000)
        elif op == Opcode.IAND:
            result = src1 & src2
            self.registers[dst] = np.int32(result)
        elif op == Opcode.IOR:
            result = src1 | src2
            self.registers[dst] = np.int32(result)
        elif op == Opcode.IXOR:
            result = src1 ^ src2
            self.registers[dst] = np.int32(result)
        elif op == Opcode.ISHL:
            shift = src2 & 0x1F  # Only bottom 5 bits
            result = (src1 << shift) & 0xFFFFFFFF
            self.registers[dst] = np.int32(result if result < 0x80000000 else result - 0x100000000)
        elif op == Opcode.ISHR:
            shift = src2 & 0x1F
            result = src1 >> shift
            self.registers[dst] = np.int32(result)
        elif op == Opcode.IMOV:
            result = src1
            self.registers[dst] = np.int32(src1)

        # Floating-point operations
        elif op == Opcode.FADD:
            f1 = self._int_to_float(src1)
            f2 = self._int_to_float(src2)
            result = self._float_to_int(f1 + f2)
            self.registers[dst] = np.int32(result)
        elif op == Opcode.FSUB:
            f1 = self._int_to_float(src1)
            f2 = self._int_to_float(src2)
            result = self._float_to_int(f1 - f2)
            self.registers[dst] = np.int32(result)
        elif op == Opcode.FMUL:
            f1 = self._int_to_float(src1)
            f2 = self._int_to_float(src2)
            result = self._float_to_int(f1 * f2)
            self.registers[dst] = np.int32(result)
        elif op == Opcode.FFMA:
            f1 = self._int_to_float(src1)
            f2 = self._int_to_float(src2)
            f3 = self._int_to_float(int(self.registers[instr.src3]))
            result = self._float_to_int(f1 * f2 + f3)
            self.registers[dst] = np.int32(result)

        # Memory operations
        elif op == Opcode.LD:
            result = self._load_word(instr.immediate)
            self.registers[dst] = np.int32(result)
        elif op == Opcode.ST:
            self._store_word(instr.immediate, src1)
        elif op == Opcode.LDI:
            result = instr.immediate
            self.registers[dst] = np.int32(instr.immediate)

        # Comparison operations (set predicate)
        elif op == Opcode.ICMP_EQ:
            self.predicate = src1 == src2
        elif op == Opcode.ICMP_NE:
            self.predicate = src1 != src2
        elif op == Opcode.ICMP_LT:
            self.predicate = src1 < src2
        elif op == Opcode.ICMP_LE:
            self.predicate = src1 <= src2
        elif op == Opcode.ICMP_GT:
            self.predicate = src1 > src2
        elif op == Opcode.ICMP_GE:
            self.predicate = src1 >= src2

        # Mask control
        elif op == Opcode.SETMASK:
            self.active = self.predicate
        elif op == Opcode.CLRMASK:
            self.active = True

        return result

    def _load_word(self, addr: int) -> int:
        """Load 32-bit word from local memory."""
        if addr + 3 >= len(self.local_memory):
            return 0
        # Little-endian load
        b0 = int(self.local_memory[addr])
        b1 = int(self.local_memory[addr + 1])
        b2 = int(self.local_memory[addr + 2])
        b3 = int(self.local_memory[addr + 3])
        value = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        # Sign-extend if negative
        if value >= 0x80000000:
            value -= 0x100000000
        return value

    def _store_word(self, addr: int, value: int) -> None:
        """Store 32-bit word to local memory."""
        if addr + 3 >= len(self.local_memory):
            return
        # Little-endian store
        value = value & 0xFFFFFFFF
        self.local_memory[addr] = value & 0xFF
        self.local_memory[addr + 1] = (value >> 8) & 0xFF
        self.local_memory[addr + 2] = (value >> 16) & 0xFF
        self.local_memory[addr + 3] = (value >> 24) & 0xFF

    def _int_to_float(self, bits: int) -> float:
        """Convert 32-bit integer to float (IEEE 754 interpretation)."""
        import struct

        # Handle negative values
        if bits < 0:
            bits = bits & 0xFFFFFFFF
        return struct.unpack("f", struct.pack("I", bits))[0]

    def _float_to_int(self, f: float) -> int:
        """Convert float to 32-bit integer (IEEE 754 bits)."""
        import struct

        bits = struct.unpack("I", struct.pack("f", f))[0]
        if bits >= 0x80000000:
            return bits - 0x100000000
        return bits

    def get_register(self, idx: int) -> int:
        """Get register value as Python int."""
        return int(self.registers[idx])

    def set_register(self, idx: int, value: int) -> None:
        """Set register value."""
        self.registers[idx] = np.int32(value & 0xFFFFFFFF)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"PE({self.row}, {self.col}, active={self.active})"
