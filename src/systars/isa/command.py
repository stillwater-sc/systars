"""
The command port shared by the ISA instructions (``Matmul``, ``Conv2d``).

An ISA instruction issues the same commands a host would: each names its target
controller (``cmd_type``, the ``SystolicTop.CTRL_*`` values) and that
controller's own opcode (``DmaOpcode`` for LOAD/STORE, ``ExecOpcode`` for
EXECUTE), with the fields ``SystolicTop`` takes. ``connect_to_top`` wires an
instruction straight to a ``SystolicTop``.

``SystolicTop`` has no dependency tracking (a COMPUTE could start before the
LOAD of its operands finished), so instructions issue one command at a time: a
command is held until ``cmd_ready``, and the next is not issued until
``cmd_completed`` reports it done.
"""

from amaranth import Module, Signal
from amaranth.lib.wiring import In, Out

# Targets, as SystolicTop.cmd_type (kept here so the ISA does not import the top)
CTRL_LOAD = 0
CTRL_STORE = 1
CTRL_EXEC = 2

_FIELDS = {
    "dram_addr": 64,
    "local_addr": 32,
    "len": 16,
    "stride": 32,
    "row_bytes": 16,
    "rs1": 32,
    "rs2": 32,
    "rd": 32,
    "k_dim": 16,
    "activation": 4,
}


def command_ports() -> dict:
    """Signature entries of the command port."""
    ports = {
        "cmd_valid": Out(1),
        "cmd_ready": In(1),
        "cmd_type": Out(2),
        "cmd_opcode": Out(8),
        "cmd_id": Out(8),
        "cmd_completed": In(1),
    }
    ports.update({f"cmd_{name}": Out(width) for name, width in _FIELDS.items()})
    return ports


class CommandIssuer:
    """Builds FSM states that issue one command and wait for its completion."""

    def __init__(self, m: Module, isa):
        self.m, self.isa = m, isa
        self.cmd_id = Signal(8, name="cmd_id_counter")
        m.d.comb += isa.cmd_id.eq(self.cmd_id)

    def reset(self):
        """Statements that restart the command IDs (for an instruction's INIT state)."""
        return [self.cmd_id.eq(0)]

    def state(self, name: str, next_state: str, cmd_type: int, opcode, **fields):
        """Add states ``name`` (issue) and ``name_WAIT`` (wait for completion).

        ``opcode`` and the field values may be constants or Amaranth values. Must be
        called inside the instruction's ``with m.FSM()``.
        """
        m, isa = self.m, self.isa
        unknown = set(fields) - set(_FIELDS)
        if unknown:
            raise ValueError(f"unknown command fields {sorted(unknown)}")
        with m.State(name):
            m.d.comb += [isa.cmd_valid.eq(1), isa.cmd_type.eq(cmd_type), isa.cmd_opcode.eq(opcode)]
            m.d.comb += [getattr(isa, f"cmd_{k}").eq(v) for k, v in fields.items()]
            with m.If(isa.cmd_ready):
                m.next = f"{name}_WAIT"
        with m.State(f"{name}_WAIT"), m.If(isa.cmd_completed):
            m.d.sync += self.cmd_id.eq(self.cmd_id + 1)
            m.next = next_state


def connect_to_top(m: Module, isa, top) -> None:
    """Drive ``top``'s command port from an ISA instruction."""
    m.d.comb += [
        top.cmd_valid.eq(isa.cmd_valid),
        isa.cmd_ready.eq(top.cmd_ready),
        top.cmd_type.eq(isa.cmd_type),
        top.cmd_opcode.eq(isa.cmd_opcode),
        top.cmd_id.eq(isa.cmd_id),
        isa.cmd_completed.eq(top.load_completed | top.store_completed | top.exec_completed),
    ]
    m.d.comb += [getattr(top, f"cmd_{name}").eq(getattr(isa, f"cmd_{name}")) for name in _FIELDS]
