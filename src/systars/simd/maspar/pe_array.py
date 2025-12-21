"""
MasPar PE Array with XNET mesh connectivity.

The PE array is a 2D grid of Processing Elements connected via the XNET
8-neighbor mesh network. Data can flow between adjacent PEs in a single
cycle via XNET transfers.

XNET Directions:
- Cardinal: North, South, East, West
- Diagonal: NorthEast, NorthWest, SouthEast, SouthWest

The mesh uses toroidal (wrap-around) addressing, so the top row connects
to the bottom row and the left column connects to the right column.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .config import MasParConfig
from .instruction import XNET_OFFSETS, XNET_OPCODES, Instruction, Opcode, XNETDirection
from .pe import PE


@dataclass
class PEArray:
    """
    2D array of Processing Elements with XNET mesh connectivity.

    The array supports:
    - Broadcast instruction execution (all PEs execute same instruction)
    - XNET neighbor-to-neighbor data transfers
    - PE masking for conditional execution
    - Reduction operations across all PEs
    """

    config: MasParConfig

    # 2D array of PEs [row][col]
    pes: list[list[PE]] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the PE array."""
        self.pes = [
            [PE(row=r, col=c, config=self.config) for c in range(self.config.array_cols)]
            for r in range(self.config.array_rows)
        ]

    def reset(self) -> None:
        """Reset all PEs."""
        for row in self.pes:
            for pe in row:
                pe.reset()

    def get_pe(self, row: int, col: int) -> PE:
        """Get PE at specified position."""
        return self.pes[row][col]

    def broadcast_instruction(self, instr: Instruction) -> None:
        """
        Broadcast instruction to all active PEs.

        This is the core SIMD operation - every PE executes the same
        instruction on its local data.
        """
        # Handle XNET instructions specially
        if instr.opcode in XNET_OPCODES:
            direction = XNET_OPCODES[instr.opcode]
            self.xnet_shift(direction, instr.src1, instr.dst)
            return

        # Handle reduction operations
        if instr.opcode in (Opcode.REDUCE_SUM, Opcode.REDUCE_MAX, Opcode.REDUCE_MIN):
            self._execute_reduction(instr)
            return

        # Standard instruction broadcast
        for row in self.pes:
            for pe in row:
                pe.execute(instr)

    def xnet_shift(self, direction: XNETDirection, src_reg: int, dst_reg: int) -> None:
        """
        Shift data between neighbors via XNET mesh.

        Each PE reads from its neighbor in the specified direction.
        Uses toroidal wrap-around at array boundaries.

        Args:
            direction: XNET direction to shift from
            src_reg: Source register index (read from neighbor)
            dst_reg: Destination register index (write locally)
        """
        rows = self.config.array_rows
        cols = self.config.array_cols
        dr, dc = XNET_OFFSETS[direction]

        # Collect values from neighbors first (before any writes)
        new_values: list[list[int]] = []
        for r in range(rows):
            row_values = []
            for c in range(cols):
                # Find neighbor with toroidal wrap
                neighbor_r = (r + dr) % rows
                neighbor_c = (c + dc) % cols
                neighbor = self.pes[neighbor_r][neighbor_c]
                # Read neighbor's register value
                row_values.append(neighbor.get_register(src_reg))
            new_values.append(row_values)

        # Write new values to destination registers
        for r in range(rows):
            for c in range(cols):
                pe = self.pes[r][c]
                if pe.active:
                    pe.set_register(dst_reg, new_values[r][c])

    def set_mask_by_predicate(self) -> None:
        """Set PE active states based on their predicate values."""
        for row in self.pes:
            for pe in row:
                pe.active = pe.predicate

    def clear_mask(self) -> None:
        """Clear mask - all PEs become active."""
        for row in self.pes:
            for pe in row:
                pe.active = True

    def set_mask_by_function(self, predicate_fn: Callable[[PE], bool]) -> None:
        """
        Set PE active states based on a predicate function.

        Args:
            predicate_fn: Function that takes a PE and returns True if active
        """
        for row in self.pes:
            for pe in row:
                pe.active = predicate_fn(pe)

    def _execute_reduction(self, instr: Instruction) -> None:
        """
        Execute a reduction operation across all active PEs.

        The result is stored in register dst of all PEs.
        """
        values = []
        for row in self.pes:
            for pe in row:
                if pe.active:
                    values.append(pe.get_register(instr.src1))

        if not values:
            return

        if instr.opcode == Opcode.REDUCE_SUM:
            result = sum(values)
        elif instr.opcode == Opcode.REDUCE_MAX:
            result = max(values)
        elif instr.opcode == Opcode.REDUCE_MIN:
            result = min(values)
        else:
            return

        # Store result in all PEs
        for row in self.pes:
            for pe in row:
                pe.set_register(instr.dst, result)

    def load_matrix_block(
        self, matrix: np.ndarray, register: int, start_row: int = 0, start_col: int = 0
    ) -> None:
        """
        Load a matrix block into PE registers.

        Each PE(i,j) receives matrix[i,j] in the specified register.

        Args:
            matrix: 2D numpy array to load
            register: Destination register index
            start_row: Starting row offset in matrix
            start_col: Starting column offset in matrix
        """
        rows = min(matrix.shape[0], self.config.array_rows)
        cols = min(matrix.shape[1], self.config.array_cols)

        for r in range(rows):
            for c in range(cols):
                value = int(matrix[start_row + r, start_col + c])
                self.pes[r][c].set_register(register, value)

    def extract_matrix_block(
        self, register: int, rows: int | None = None, cols: int | None = None
    ) -> np.ndarray:
        """
        Extract matrix block from PE registers.

        Args:
            register: Source register index
            rows: Number of rows to extract (default: array_rows)
            cols: Number of columns to extract (default: array_cols)

        Returns:
            2D numpy array with register values from each PE
        """
        if rows is None:
            rows = self.config.array_rows
        if cols is None:
            cols = self.config.array_cols

        result = np.zeros((rows, cols), dtype=np.int32)
        for r in range(rows):
            for c in range(cols):
                result[r, c] = self.pes[r][c].get_register(register)

        return result

    def get_active_count(self) -> int:
        """Count number of active PEs."""
        count = 0
        for row in self.pes:
            for pe in row:
                if pe.active:
                    count += 1
        return count

    def dump_registers(self, register: int) -> str:
        """
        Dump register values for debugging.

        Args:
            register: Register index to dump

        Returns:
            Formatted string showing register values across the array
        """
        lines = []
        for r, row in enumerate(self.pes):
            values = [f"{pe.get_register(register):4d}" for pe in row]
            lines.append(f"Row {r:2d}: " + " ".join(values))
        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        active = self.get_active_count()
        total = self.config.total_pes
        return (
            f"PEArray({self.config.array_rows}x{self.config.array_cols}, active={active}/{total})"
        )
