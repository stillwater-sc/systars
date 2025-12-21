"""
MasPar instruction set definitions.

This module defines the opcodes and instruction format for the MasPar
SIMD array processor. Instructions are broadcast from the ACU to all PEs.

Instruction Categories:
- Integer ALU: IADD, ISUB, IMUL, IAND, IOR, IXOR, ISHL, ISHR
- Floating-point: FADD, FSUB, FMUL, FFMA
- Memory: LD, ST (local PE memory)
- Communication: XNET (neighbor shift), ROUTE (global router)
- Control: SETMASK, CLRMASK (PE masking for conditionals)
- Special: REDUCE (reduction across PEs)
"""

from dataclasses import dataclass
from enum import IntEnum


class Opcode(IntEnum):
    """MasPar instruction opcodes."""

    NOP = 0

    # Integer ALU (1 cycle, except multiply)
    IADD = 1  # Rdst = Rsrc1 + Rsrc2
    ISUB = 2  # Rdst = Rsrc1 - Rsrc2
    IMUL = 3  # Rdst = Rsrc1 * Rsrc2 (4 cycles)
    IAND = 4  # Rdst = Rsrc1 & Rsrc2
    IOR = 5  # Rdst = Rsrc1 | Rsrc2
    IXOR = 6  # Rdst = Rsrc1 ^ Rsrc2
    ISHL = 7  # Rdst = Rsrc1 << Rsrc2
    ISHR = 8  # Rdst = Rsrc1 >> Rsrc2
    IMOV = 9  # Rdst = Rsrc1

    # Floating-point (4 cycles)
    FADD = 10  # Rdst = Rsrc1 + Rsrc2
    FSUB = 11  # Rdst = Rsrc1 - Rsrc2
    FMUL = 12  # Rdst = Rsrc1 * Rsrc2
    FFMA = 13  # Rdst = Rsrc1 * Rsrc2 + Rsrc3

    # Memory (2 cycles) - local PE memory
    LD = 20  # Rdst = mem[addr]
    ST = 21  # mem[addr] = Rsrc1
    LDI = 22  # Rdst = immediate (load immediate)

    # Communication - XNET mesh
    XNET_N = 30  # Rdst = neighbor[NORTH].Rsrc1
    XNET_S = 31  # Rdst = neighbor[SOUTH].Rsrc1
    XNET_E = 32  # Rdst = neighbor[EAST].Rsrc1
    XNET_W = 33  # Rdst = neighbor[WEST].Rsrc1
    XNET_NE = 34  # Rdst = neighbor[NORTHEAST].Rsrc1
    XNET_NW = 35  # Rdst = neighbor[NORTHWEST].Rsrc1
    XNET_SE = 36  # Rdst = neighbor[SOUTHEAST].Rsrc1
    XNET_SW = 37  # Rdst = neighbor[SOUTHWEST].Rsrc1

    # Communication - Global router
    ROUTE_SEND = 40  # Send Rsrc1 to PE at (imm_row, imm_col)
    ROUTE_RECV = 41  # Rdst = receive from router

    # Control - PE masking
    SETMASK = 50  # Set PE active if condition true
    CLRMASK = 51  # Clear mask, all PEs active

    # Special operations
    REDUCE_SUM = 60  # Global sum reduction
    REDUCE_MAX = 61  # Global max reduction
    REDUCE_MIN = 62  # Global min reduction

    # Comparison (sets predicate)
    ICMP_EQ = 70  # Set predicate if Rsrc1 == Rsrc2
    ICMP_NE = 71  # Set predicate if Rsrc1 != Rsrc2
    ICMP_LT = 72  # Set predicate if Rsrc1 < Rsrc2
    ICMP_LE = 73  # Set predicate if Rsrc1 <= Rsrc2
    ICMP_GT = 74  # Set predicate if Rsrc1 > Rsrc2
    ICMP_GE = 75  # Set predicate if Rsrc1 >= Rsrc2


# XNET direction constants for convenience
class XNETDirection(IntEnum):
    """XNET neighbor directions."""

    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    NORTHEAST = 4
    NORTHWEST = 5
    SOUTHEAST = 6
    SOUTHWEST = 7


# Direction offsets (row_delta, col_delta)
XNET_OFFSETS = {
    XNETDirection.NORTH: (-1, 0),
    XNETDirection.SOUTH: (1, 0),
    XNETDirection.EAST: (0, 1),
    XNETDirection.WEST: (0, -1),
    XNETDirection.NORTHEAST: (-1, 1),
    XNETDirection.NORTHWEST: (-1, -1),
    XNETDirection.SOUTHEAST: (1, 1),
    XNETDirection.SOUTHWEST: (1, -1),
}

# Opcode to XNET direction mapping
XNET_OPCODES = {
    Opcode.XNET_N: XNETDirection.NORTH,
    Opcode.XNET_S: XNETDirection.SOUTH,
    Opcode.XNET_E: XNETDirection.EAST,
    Opcode.XNET_W: XNETDirection.WEST,
    Opcode.XNET_NE: XNETDirection.NORTHEAST,
    Opcode.XNET_NW: XNETDirection.NORTHWEST,
    Opcode.XNET_SE: XNETDirection.SOUTHEAST,
    Opcode.XNET_SW: XNETDirection.SOUTHWEST,
}

# Opcode latencies (cycles)
OPCODE_LATENCY = {
    Opcode.NOP: 1,
    # Integer ALU
    Opcode.IADD: 1,
    Opcode.ISUB: 1,
    Opcode.IMUL: 4,
    Opcode.IAND: 1,
    Opcode.IOR: 1,
    Opcode.IXOR: 1,
    Opcode.ISHL: 1,
    Opcode.ISHR: 1,
    Opcode.IMOV: 1,
    # Floating-point
    Opcode.FADD: 4,
    Opcode.FSUB: 4,
    Opcode.FMUL: 4,
    Opcode.FFMA: 4,
    # Memory
    Opcode.LD: 2,
    Opcode.ST: 2,
    Opcode.LDI: 1,
    # XNET
    Opcode.XNET_N: 1,
    Opcode.XNET_S: 1,
    Opcode.XNET_E: 1,
    Opcode.XNET_W: 1,
    Opcode.XNET_NE: 1,
    Opcode.XNET_NW: 1,
    Opcode.XNET_SE: 1,
    Opcode.XNET_SW: 1,
    # Router (much slower)
    Opcode.ROUTE_SEND: 10,
    Opcode.ROUTE_RECV: 10,
    # Control
    Opcode.SETMASK: 1,
    Opcode.CLRMASK: 1,
    # Reductions (depends on array size)
    Opcode.REDUCE_SUM: 10,
    Opcode.REDUCE_MAX: 10,
    Opcode.REDUCE_MIN: 10,
    # Comparison
    Opcode.ICMP_EQ: 1,
    Opcode.ICMP_NE: 1,
    Opcode.ICMP_LT: 1,
    Opcode.ICMP_LE: 1,
    Opcode.ICMP_GT: 1,
    Opcode.ICMP_GE: 1,
}


@dataclass
class Instruction:
    """
    MasPar instruction.

    Instructions are broadcast from the ACU to all PEs. Each PE executes
    the same instruction on its local data (SIMD model).

    Attributes:
        opcode: Operation to perform
        dst: Destination register index
        src1: Source register 1 index
        src2: Source register 2 index (or immediate for some ops)
        src3: Source register 3 index (for FMA)
        immediate: Immediate value (address, constant, etc.)
    """

    opcode: Opcode = Opcode.NOP
    dst: int = 0
    src1: int = 0
    src2: int = 0
    src3: int = 0
    immediate: int = 0

    @property
    def latency(self) -> int:
        """Get instruction latency in cycles."""
        return OPCODE_LATENCY.get(self.opcode, 1)

    def is_xnet(self) -> bool:
        """Check if this is an XNET communication instruction."""
        return self.opcode in XNET_OPCODES

    def is_memory(self) -> bool:
        """Check if this is a memory instruction."""
        return self.opcode in (Opcode.LD, Opcode.ST, Opcode.LDI)

    def is_alu(self) -> bool:
        """Check if this is an ALU instruction."""
        return self.opcode in (
            Opcode.IADD,
            Opcode.ISUB,
            Opcode.IMUL,
            Opcode.IAND,
            Opcode.IOR,
            Opcode.IXOR,
            Opcode.ISHL,
            Opcode.ISHR,
            Opcode.IMOV,
        )

    def is_fpu(self) -> bool:
        """Check if this is a floating-point instruction."""
        return self.opcode in (Opcode.FADD, Opcode.FSUB, Opcode.FMUL, Opcode.FFMA)

    def __str__(self) -> str:
        """Format instruction as assembly-like string."""
        op_name = self.opcode.name

        if self.opcode == Opcode.NOP:
            return "NOP"
        elif self.opcode == Opcode.LDI:
            return f"LDI R{self.dst}, {self.immediate}"
        elif self.opcode in (Opcode.LD,):
            return f"LD R{self.dst}, [{self.immediate}]"
        elif self.opcode == Opcode.ST:
            return f"ST [{self.immediate}], R{self.src1}"
        elif self.opcode == Opcode.FFMA:
            return f"FFMA R{self.dst}, R{self.src1}, R{self.src2}, R{self.src3}"
        elif self.opcode in XNET_OPCODES:
            return f"{op_name} R{self.dst}, R{self.src1}"
        elif self.opcode == Opcode.IMOV:
            return f"MOV R{self.dst}, R{self.src1}"
        else:
            return f"{op_name} R{self.dst}, R{self.src1}, R{self.src2}"
