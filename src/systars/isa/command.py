"""
The command port shared by the ISA instructions (``Matmul``, ``Conv2d``).

An ISA instruction issues the same commands a host would: each names its target
process (``cmd_type``, the ``SystolicTop.CTRL_*`` values) and that process's own
opcode (``DmaOpcode`` for LOAD/STORE, ``ExecOpcode`` for EXECUTE), with the fields
``SystolicTop`` takes. ``connect_to_top`` wires an instruction straight to a
``SystolicTop``.

Commands are pushed, never waited on: ``SystolicTop`` queues each process's
commands, and credits and tile-ready tokens (``cmd_sync``, see
``controller/sync.py``) order them. An instruction issues a command as soon as its
process's queue has room (``cmd_ready``), and is done once every STORE it issued
has completed (``cmd_store_completed``): its results are in L3.

``TileBuffers`` is the instruction's buffer schedule: ``buffers`` A tile slots and
``buffers`` B tile slots in L2, and ``buffers`` accumulator slots, used in ring
order. Two buffers let the BlockMover fill the next tile's slot while the Streamer
works on the current one; one serializes them. An OS COMPUTE reads both tiles of
its pair, so OS tracks the pair with the A slot's credit alone: the A tile's LOAD
takes it and the B tile's LOAD emits its token.

Issuing in program order cannot deadlock: every credit a command waits for is
returned by a command issued before it.
"""

from amaranth import Cat, Module, Mux, Signal
from amaranth.lib.wiring import In, Out

from ..config import SystolicConfig
from ..controller.sync import Sync
from ..memory.local_addr import LocalAddr

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
    "chain": 2,
    "sync": Sync.BITS,
}


def command_ports() -> dict:
    """Signature entries of the command port."""
    ports = {
        "cmd_valid": Out(1),
        "cmd_ready": In(1),
        "cmd_type": Out(2),
        "cmd_opcode": Out(8),
        "cmd_id": Out(8),
        "cmd_store_completed": In(1),
    }
    ports.update({f"cmd_{name}": Out(width) for name, width in _FIELDS.items()})
    return ports


class CommandIssuer:
    """Builds FSM states that push one command each, and tracks the STOREs in flight."""

    def __init__(self, m: Module, isa):
        self.m, self.isa = m, isa
        self.cmd_id = Signal(8, name="cmd_id_counter")
        self.stores_issued = Signal(32, name="stores_issued")
        self.stores_done = Signal(32, name="stores_done")
        m.d.comb += isa.cmd_id.eq(self.cmd_id)
        with m.If(isa.cmd_store_completed):
            m.d.sync += self.stores_done.eq(self.stores_done + 1)

    def reset(self):
        """Statements that restart the command IDs (for an instruction's START)."""
        return [self.cmd_id.eq(0), self.stores_issued.eq(0), self.stores_done.eq(0)]

    def drained(self):
        """Every STORE issued has completed."""
        return self.stores_done == self.stores_issued

    def state(self, name, next_state, cmd_type, opcode, *, then=(), **fields):
        """Add state ``name``: push one command, then go to ``next_state``.

        ``opcode`` and the field values may be constants or Amaranth values; ``then``
        are statements to run when the command is accepted. Must be called inside the
        instruction's ``with m.FSM()``.
        """
        m, isa = self.m, self.isa
        unknown = set(fields) - set(_FIELDS)
        if unknown:
            raise ValueError(f"unknown command fields {sorted(unknown)}")
        with m.State(name):
            m.d.comb += [isa.cmd_valid.eq(1), isa.cmd_type.eq(cmd_type), isa.cmd_opcode.eq(opcode)]
            m.d.comb += [getattr(isa, f"cmd_{k}").eq(v) for k, v in fields.items()]
            with m.If(isa.cmd_ready):
                m.d.sync += [self.cmd_id.eq(self.cmd_id + 1), *then]
                if cmd_type == CTRL_STORE:
                    m.d.sync += self.stores_issued.eq(self.stores_issued + 1)
                m.next = next_state


class TileBuffers:
    """Ring-ordered tile slots for an instruction on a ``dim x dim`` array.

    L2 slot ``n`` (A) is scratchpad rows ``[n*dim, (n+1)*dim)``; L2 slot
    ``buffers + n`` (B) is the ``n``-th tile after the A tiles; accumulator slot
    ``n`` is accumulator rows ``[n*dim, (n+1)*dim)``. Scratchpad and accumulator
    rows are linear in the local address (``LocalAddr``: the bank is the low bits).
    """

    def __init__(self, config: SystolicConfig, dim: int, buffers: int, owner: str):
        if buffers < 1:
            raise ValueError(f"{owner}: buffers must be at least 1, got {buffers}")
        if config.l2_slots < 2 * buffers or config.acc_slots < buffers:
            raise ValueError(
                f"{owner} with {buffers} buffers needs {2 * buffers} L2 slots and "
                f"{buffers} accumulator slots (config has {config.l2_slots} and "
                f"{config.acc_slots})"
            )
        if config.sp_total_rows < 2 * buffers * dim or config.acc_total_rows < buffers * dim:
            raise ValueError(
                f"{owner} with {buffers} buffers needs {2 * buffers * dim} scratchpad rows "
                f"and {buffers * dim} accumulator rows"
            )
        self.dim, self.buffers = dim, buffers
        self.acc_flag = LocalAddr(config).make_acc_addr(0, 0)

    def build(self, m: Module):
        """Create the ring counters, slot tags and slot addresses."""
        dim, buffers = self.dim, self.buffers
        self.l2_n = Signal(range(buffers), name="l2_buffer")  # the A/B pair in use
        self.acc_n = Signal(range(buffers), name="acc_buffer")
        self.a_slot = Signal(4, name="a_slot")
        self.b_slot = Signal(4, name="b_slot")
        self.acc_slot = Signal(4, name="acc_slot")
        self.a_addr = Signal(32, name="sp_a_addr")
        self.b_addr = Signal(32, name="sp_b_addr")
        self.acc_addr = Signal(32, name="acc_c_addr")
        m.d.comb += [
            self.a_slot.eq(self.l2_n),
            self.b_slot.eq(buffers + self.l2_n),
            self.acc_slot.eq(self.acc_n),
            self.a_addr.eq(self.l2_n * dim),
            self.b_addr.eq((buffers + self.l2_n) * dim),
            self.acc_addr.eq(self.acc_flag | (self.acc_n * dim)),
        ]
        return self

    def reset(self):
        return [self.l2_n.eq(0), self.acc_n.eq(0)]

    def next_l2(self):
        """Statements that move on to the next A/B slot pair."""
        return [self.l2_n.eq(Mux(self.l2_n == self.buffers - 1, 0, self.l2_n + 1))]

    def next_acc(self):
        """Statements that move on to the next accumulator slot."""
        return [self.acc_n.eq(Mux(self.acc_n == self.buffers - 1, 0, self.acc_n + 1))]

    # cmd_sync values of the commands a K tile issues

    @staticmethod
    def load(slot, *, first=True, last=True):
        """LOAD into ``slot``: the credit on the tile's first command, the token on its last."""
        return Sync.encode(dst=slot, acquire=first, signal=last)

    def preload(self, slot=None):
        """PRELOAD: waits for the stationary tile (default: B), returns its credit once
        it is in the fabric."""
        slot = self.b_slot if slot is None else slot
        return Sync.encode(src=slot, wait=True, release=True)

    def compute(self, first_k, last_k, slot=None):
        """COMPUTE: consumes the streamed tile (default: A); fills the accumulator
        tile across the K steps."""
        slot = self.a_slot if slot is None else slot
        return Sync.encode(
            src=slot, wait=True, release=True,
            dst=self.acc_slot, acquire=first_k, signal=last_k,
        )  # fmt: skip

    @staticmethod
    def chain(first_k, last_k):
        """``cmd_chain`` of an OS COMPUTE: one reduction across the K tiles."""
        return Cat(~first_k, ~last_k)

    def store(self):
        """STORE: waits for the result token, returns the accumulator credit."""
        return Sync.encode(src=self.acc_slot, wait=True, release=True)


def connect_to_top(m: Module, isa, top) -> None:
    """Drive ``top``'s command port from an ISA instruction."""
    m.d.comb += [
        top.cmd_valid.eq(isa.cmd_valid),
        isa.cmd_ready.eq(top.cmd_ready),
        top.cmd_type.eq(isa.cmd_type),
        top.cmd_opcode.eq(isa.cmd_opcode),
        top.cmd_id.eq(isa.cmd_id),
        isa.cmd_store_completed.eq(top.store_completed),
    ]
    m.d.comb += [getattr(top, f"cmd_{name}").eq(getattr(isa, f"cmd_{name}")) for name in _FIELDS]
