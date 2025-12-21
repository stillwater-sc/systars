"""
MasPar Array Control Unit (ACU).

The ACU is responsible for:
- Instruction fetch from program memory
- Instruction decode
- Instruction broadcast to all PEs
- Scalar register management for loop control
- Program counter management

The ACU is the "scalar" processor that orchestrates the parallel PE array.
It issues one instruction per cycle that all PEs execute simultaneously.
"""

from dataclasses import dataclass, field

from .instruction import OPCODE_LATENCY, Instruction


@dataclass
class ACU:
    """
    Array Control Unit for MasPar SIMD processor.

    The ACU fetches, decodes, and broadcasts instructions to the PE array.
    It maintains the program counter and scalar registers for loop control.
    """

    # Program storage
    program: list[Instruction] = field(default_factory=list)

    # Program counter
    pc: int = 0

    # Scalar registers for loop control (not PE registers)
    scalar_registers: list[int] = field(default_factory=lambda: [0] * 16)

    # Currently executing instruction and remaining cycles
    current_instruction: Instruction | None = None
    remaining_cycles: int = 0

    def reset(self) -> None:
        """Reset ACU state."""
        self.pc = 0
        self.current_instruction = None
        self.remaining_cycles = 0
        self.scalar_registers = [0] * 16

    def load_program(self, program: list[Instruction]) -> None:
        """
        Load a program into the ACU.

        Args:
            program: List of instructions to execute
        """
        self.program = program
        self.pc = 0
        self.current_instruction = None
        self.remaining_cycles = 0

    def fetch(self) -> Instruction | None:
        """
        Fetch the next instruction.

        If a multi-cycle instruction is in progress, returns None
        until it completes.

        Returns:
            Next instruction to execute, or None if done/waiting
        """
        # If still executing multi-cycle instruction
        if self.remaining_cycles > 0:
            self.remaining_cycles -= 1
            return None

        # Check if program complete
        if self.pc >= len(self.program):
            return None

        # Fetch next instruction
        instr = self.program[self.pc]
        self.pc += 1

        # Set up multi-cycle execution if needed
        latency = OPCODE_LATENCY.get(instr.opcode, 1)
        if latency > 1:
            self.current_instruction = instr
            self.remaining_cycles = latency - 1

        return instr

    def is_done(self) -> bool:
        """Check if program execution is complete."""
        return self.pc >= len(self.program) and self.remaining_cycles == 0

    def is_stalled(self) -> bool:
        """Check if ACU is stalled on a multi-cycle instruction."""
        return self.remaining_cycles > 0

    def get_progress(self) -> tuple[int, int]:
        """
        Get program execution progress.

        Returns:
            Tuple of (current_pc, total_instructions)
        """
        return (self.pc, len(self.program))

    def set_scalar(self, idx: int, value: int) -> None:
        """Set scalar register value."""
        if 0 <= idx < len(self.scalar_registers):
            self.scalar_registers[idx] = value

    def get_scalar(self, idx: int) -> int:
        """Get scalar register value."""
        if 0 <= idx < len(self.scalar_registers):
            return self.scalar_registers[idx]
        return 0

    def __repr__(self) -> str:
        """String representation."""
        pc, total = self.get_progress()
        status = "done" if self.is_done() else "stalled" if self.is_stalled() else "running"
        return f"ACU(pc={pc}/{total}, {status})"
