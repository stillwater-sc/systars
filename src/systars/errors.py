"""
The systolic tile's first-error capture (#84): what each sticky error bit's monitor reports,
and how the tile packs the first error into ``ERR_INFO``.

The tile's error bits are sticky and say only that something went wrong. Each monitor also
reports **what kind** of error it saw first (its ``error_kind`` output, stable from the clock
its ``error`` bit rises), and ``SystolicTop`` latches the first error event of all of them:

- the **source**: L1 (the Streamer's schedule and the L1 buffers), L2 (an access the
  macros can't take), SYNC (a credit or token protocol error, in one of the two slot pools)
  or AXI (an error response);
- the **kind**, from that source's table below;
- the **command** (its type and ID), when the kind names a controller:
  - a SYNC fault at a command's completion (a signal or a release,
    ``SlotPool.error_at_completion``) names the command that completed, though a later one
    may have started since or wait at the queue head;
  - a slot number past the pool at a start or a head check comes from a command that can
    never start, so it names the command at the head of that controller's queue;
  - every other kind names the command the controller started last, once it has started
    one. The Streamer runs commands back to back, so for an L1 or L2 error the named
    command can be the one after the one at fault;
- the **cycle** it was seen (``ERR_CYCLE``, from the free-running ``CYCLE`` counter), and
  whether another source raised an error in the same clock (``MULTI``).

The capture holds the first error since reset, or since ``ERR_INFO`` was last written: a
write re-arms it. The L1, L2 and SYNC bits stay set until reset, so a re-armed capture
records only a new event, an AXI error or a source that had not raised its bit yet.

``ERR_INFO``: ``VALID[0] SOURCE[3:1] KIND[11:4] CMD_VALID[12] CMD_TYPE[14:13] MULTI[15]
CMD_ID[23:16]``. ``FirstError.decode`` turns it into names.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from amaranth import Module, Mux, Signal
from amaranth.lib.wiring import Component, In, Out


class ErrorSource(IntEnum):
    NONE = 0
    L1 = 1  # STATUS_L1_ERROR: controller/streamer.py and its L1 buffers
    L2 = 2  # STATUS_L2_ERROR: memory/scratchpad.py
    SYNC = 3  # STATUS_SYNC_ERROR: controller/sync.py, the L2 or the Combiner slot pool
    AXI = 4  # STATUS_AXI_ERROR: an error response (ERROR, ERROR_ADDR)


class L1Error(IntEnum):
    """``Streamer.error_kind``."""

    SIGNAL_NO_RESULT = 1  # a command with SIGNAL that writes no result
    RESULT_WITHOUT_COMMAND = 2  # a result row with no command waiting for it
    NORTH_COLLISION = 3  # two OS results met on a north link
    A_LOAD_WITHOUT_CREDIT = 4  # an A stream buffer loaded while still streaming
    B_LOAD_WITHOUT_CREDIT = 5
    W_LOAD_WITHOUT_CREDIT = 6  # a weight (WS/IS shadow) buffer
    NORTH_RESULT_OVERFLOW = 7  # an OS result buffer written full
    SOUTH_RESULT_OVERFLOW = 8  # a WS/IS result buffer written full


class L2Error(IntEnum):
    """``Scratchpad.error_kind``."""

    READ_A_PAST_L2 = 1  # read port A addressed a row past L2
    READ_B_PAST_L2 = 2
    READS_SAME_GROUP = 3  # both reads of a clock in one bank group
    WRITE_PAST_L2 = 4
    READ_BANK_BUSY = 5  # a read of a bank still busy (l2_bank_occupancy > 1)
    WRITE_BANK_BUSY = 6


class SlotError(IntEnum):
    """``SlotPool.error_kind``: which port broke the protocol, and how."""

    TAKE_NOT_FREE = 1  # the producer took a slot that was not FREE
    SIGNAL_NOT_OWNED = 2  # the producer signalled a slot it did not own
    FILL_TAKE_NOT_FREE = 3  # the second producer (fill), the same
    FILL_SIGNAL_NOT_OWNED = 4
    DOUBLE_SIGNAL = 5  # both producers signalled one slot in a clock
    WAIT_NOT_READY = 6  # the consumer took a slot that was not READY
    RELEASE_NOT_HELD = 7  # the consumer released a slot it had not taken
    PROD_NO_SUCH_SLOT = 8  # a slot number past the pool
    FILL_NO_SUCH_SLOT = 9
    CONS_NO_SUCH_SLOT = 10
    TAKE_AND_RELEASE = 11  # the consumer took and released one slot in a clock


class SlotPoolId(IntEnum):
    """Which pool a SYNC error came from: bit 4 of its kind."""

    L2 = 0  # producer: the BlockMover's LOAD; consumer: the Streamer
    COMBINER = 1  # producer: the Streamer; fill: a LOAD into the Combiner; consumer: STORE


SYNC_POOL_SHIFT = 4


def sync_kind(pool: int, error: int) -> int:
    """A SYNC error's kind: the pool in bit 4, the ``SlotError`` below it."""
    return pool << SYNC_POOL_SHIFT | error


def axi_kind(write: bool, resp: int) -> int:
    """An AXI error's kind: 1 a read, 2 a write, and the response (2 SLVERR, 3 DECERR) in
    bits 3:2."""
    return (2 if write else 1) | resp << 2


# ERR_INFO fields: (shift, width)
VALID = (0, 1)
SOURCE = (1, 3)
KIND = (4, 8)
CMD_VALID = (12, 1)
CMD_TYPE = (13, 2)
MULTI = (15, 1)
CMD_ID = (16, 8)
KIND_BITS = KIND[1]


def _field(word: int, f: tuple[int, int]) -> int:
    return word >> f[0] & ((1 << f[1]) - 1)


def pack(
    source: int, kind: int, *, cmd_type: int | None = None, cmd_id: int = 0, multi: bool = False
) -> int:
    """An ``ERR_INFO`` word (the model the RTL is checked against)."""
    word = 1 | source << SOURCE[0] | kind << KIND[0] | int(multi) << MULTI[0]
    if cmd_type is not None:
        word |= 1 << CMD_VALID[0] | cmd_type << CMD_TYPE[0] | (cmd_id & 0xFF) << CMD_ID[0]
    return word


_COMMAND_TYPES = ("LOAD", "STORE", "EXECUTE")


def kind_name(source: int, kind: int) -> str:
    """A kind's name, e.g. ``READS_SAME_GROUP`` or ``COMBINER.WAIT_NOT_READY``."""
    try:
        if source == ErrorSource.L1:
            return L1Error(kind).name
        if source == ErrorSource.L2:
            return L2Error(kind).name
        if source == ErrorSource.SYNC:
            pool = SlotPoolId(kind >> SYNC_POOL_SHIFT).name
            return f"{pool}.{SlotError(kind & ((1 << SYNC_POOL_SHIFT) - 1)).name}"
        if source == ErrorSource.AXI:
            rw = {1: "READ", 2: "WRITE"}[kind & 3]
            return f"{rw}_{ {2: 'SLVERR', 3: 'DECERR'}.get(kind >> 2, 'OKAY') }"
    except (ValueError, KeyError):
        pass
    return f"kind {kind}"


@dataclass(frozen=True)
class FirstError:
    """A decoded first error."""

    source: ErrorSource
    kind: int
    cycle: int
    cmd_type: int | None = None  # 0 LOAD, 1 STORE, 2 EXECUTE; None: no command named
    cmd_id: int = 0
    multi: bool = False

    @classmethod
    def decode(cls, info: int, cycle: int) -> FirstError | None:
        """``ERR_INFO`` and ``ERR_CYCLE``; None when nothing was captured."""
        if not _field(info, VALID):
            return None
        has_cmd = bool(_field(info, CMD_VALID))
        return cls(
            source=ErrorSource(_field(info, SOURCE)),
            kind=_field(info, KIND),
            cycle=cycle,
            cmd_type=_field(info, CMD_TYPE) if has_cmd else None,
            cmd_id=_field(info, CMD_ID),
            multi=bool(_field(info, MULTI)),
        )

    @property
    def kind_name(self) -> str:
        return kind_name(self.source, self.kind)

    def __str__(self) -> str:
        text = f"{self.source.name} {self.kind_name} at cycle {self.cycle}"
        if self.cmd_type is not None:
            text += f", {_COMMAND_TYPES[self.cmd_type]} command {self.cmd_id}"
        if self.multi:
            text += " (another source in the same clock)"
        return text


class FirstErrorKind(Component):
    """Several sticky error sources, combined: ``error`` is any of them, and ``kind`` the
    kind of the first to rise, held from that clock on. Each source's ``kind_i`` must be
    valid from the clock its ``error_i`` rises (a constant, or a kind latched with it). In
    a clock where several rise together, the lowest index wins.

    Ports: ``error_i``, ``kind_i`` (In, per source); ``error``, ``kind`` (Out).
    """

    def __init__(self, sources: int, kind_bits: int):
        self.sources = sources
        ports = {"error": Out(1), "kind": Out(kind_bits)}
        for i in range(sources):
            ports[f"error_{i}"] = In(1)
            ports[f"kind_{i}"] = In(kind_bits)
        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()
        any_error = Signal()
        kind_now = Signal.like(self.kind)
        for i in reversed(range(self.sources)):  # the lowest index is assigned last: it wins
            with m.If(getattr(self, f"error_{i}")):
                m.d.comb += [any_error.eq(1), kind_now.eq(getattr(self, f"kind_{i}"))]
        seen = Signal()
        first = Signal.like(self.kind)
        m.d.sync += seen.eq(any_error)
        with m.If(any_error & ~seen):
            m.d.sync += first.eq(kind_now)
        m.d.comb += [self.error.eq(any_error), self.kind.eq(Mux(seen, first, kind_now))]
        return m
