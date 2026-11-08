"""
Credits and tile-ready tokens between the tile's CSP processes.

In the KPU's CSP model (kpu-sim, docs/01-architecture/kpu-execution-model.md;
the reference is ``include/sw/kpu/timing/{credit_pool,tag_cam}.hpp``) a process
pushes a tile downstream only when the downstream buffer has a free slot (a
credit), and a consumer starts only when the tile it needs has arrived (a
tile-ready token). Nobody polls and nobody requests data.

L2 (the scratchpad) and the accumulator hold whole tiles. Each tile slot has one
credit; there are no row or burst credits, since the BlockMover sequences its own
bursts inside a tile. A slot is a tag: commands still carry the slot's address
explicitly, and the issuer of the commands (an ISA instruction, or the host) is the
scheduler that assigns slots.

A slot is always in exactly one state, so credits plus occupied slots always equal
capacity:

- ``FREE``: its credit is available;
- ``OWNED``: a producer took the credit and is filling the slot;
- ``READY``: the tile is complete, and its token is waiting for the consumer;
- ``READING``: a consumer took the token and is reading the tile;
- ``LOADED`` (accumulator slots only): the BlockMover loaded the tile's starting
  values (a bias, or the C being accumulated onto), and its hand-off token is
  waiting for the Streamer, which accumulates onto them.

The consumer takes the token when it starts, so a later consumer of the same slot
(the next tile filled into it) waits for the next token even while the Streamer
still has the current one's command in flight.

Every command carries a ``cmd_sync`` field (``Sync``):

========  =============================================================
field     meaning
========  =============================================================
src       slot the command reads (consumer side)
dst       slot the command fills (producer side)
WAIT      start only when ``src`` is READY (its tile-ready token); with
          RELEASE, take the token at the start (READY -> READING)
RELEASE   on completion, return ``src``'s credit (READING -> FREE)
ACQUIRE   start only when ``dst`` is FREE, and take its credit (-> OWNED)
SIGNAL    on completion, emit ``dst``'s token (OWNED -> READY); from the
          BlockMover into an accumulator slot, the hand-off token
          (OWNED -> LOADED)
RESUME    with ACQUIRE: start only when ``dst`` is LOADED (its hand-off
          token), and take it (LOADED -> OWNED), instead of a credit
========  =============================================================

A tile moved by one command uses ACQUIRE and SIGNAL together; a tile moved by
several commands takes the credit on the first and emits the token on the last.
``cmd_sync = 0`` is an untracked command.

Pools, and who uses which side:

- L2 slots: the BlockMover's LOAD produces (dst), the Streamer's PRELOAD and
  COMPUTE consume (src).
- Accumulator slots: the Streamer's COMPUTE produces (dst; ACQUIRE on the first
  reduction step of a C tile, SIGNAL on the last), the BlockMover's STORE
  consumes (src). The token is emitted on the Streamer's result event, so SIGNAL
  belongs on a command that writes the tile's results; the Streamer flags one
  that writes none.
- A tile that starts from loaded values has two producers in turn. The
  BlockMover's accumulator LOAD takes the slot's credit (ACQUIRE) and hands the
  tile on (SIGNAL: OWNED -> LOADED); the Streamer's first COMPUTE takes the
  hand-off token (ACQUIRE with RESUME: LOADED -> OWNED) and accumulates onto the
  tile, and its last emits the result token as before. A STORE waits for READY,
  so it never takes the hand-off token for the result.

A WAIT without RELEASE only checks the token (the slot stays READY for another
reader), and a RELEASE without WAIT returns a READY slot's credit directly.

A transition from the wrong state (a SIGNAL on a slot that is not OWNED, a RELEASE
on one that is neither READING nor READY), or a slot tag outside the pool, sets the sticky
``error``. A command never takes a credit it does not have, so an out-of-range tag
stalls its queue; ``error`` reports why.
"""

from typing import NamedTuple

from amaranth import Cat, Const, Module, Mux, Signal, Value
from amaranth.lib.wiring import Component, In, Out

SLOT_BITS = 4  # Slot tags are 4 bits: up to 16 slots per pool


class SyncFields(NamedTuple):
    """The decoded fields of a ``cmd_sync`` value."""

    src: Value
    dst: Value
    wait: Value
    release: Value
    acquire: Value
    signal: Value
    resume: Value


class Sync:
    """Bit layout of ``cmd_sync``."""

    BITS = 16
    SRC = 0  # [3:0] slot read
    DST = 4  # [7:4] slot filled
    WAIT = 1 << 8
    RELEASE = 1 << 9
    ACQUIRE = 1 << 10
    SIGNAL = 1 << 11
    RESUME = 1 << 12

    @staticmethod
    def encode(
        *, src=0, dst=0, wait=False, release=False, acquire=False, signal=False, resume=False
    ):
        """``cmd_sync`` as an int, or as an Amaranth value if any argument is one."""
        args = (src, dst, wait, release, acquire, signal, resume)
        if not any(isinstance(a, Value) for a in args):
            if not (0 <= src < 16 and 0 <= dst < 16):
                raise ValueError(f"slot tags are 4 bits, got src={src} dst={dst}")
            return (
                src
                | dst << Sync.DST
                | Sync.WAIT * bool(wait)
                | Sync.RELEASE * bool(release)
                | Sync.ACQUIRE * bool(acquire)
                | Sync.SIGNAL * bool(signal)
                | Sync.RESUME * bool(resume)
            )

        def field(v, width):
            return v[:width] if isinstance(v, Value) else Const(int(v), width)

        return Cat(
            field(src, SLOT_BITS),
            field(dst, SLOT_BITS),
            field(wait, 1),
            field(release, 1),
            field(acquire, 1),
            field(signal, 1),
            field(resume, 1),
            Const(0, 3),
        )

    @staticmethod
    def fields(sync) -> SyncFields:
        """Decode a ``cmd_sync`` value."""
        return SyncFields(sync[0:4], sync[4:8], sync[8], sync[9], sync[10], sync[11], sync[12])


class SlotState:
    FREE = 0
    OWNED = 1
    READY = 2
    READING = 3
    LOADED = 4  # accumulator slots: starting values loaded, for the Streamer


class SlotPool(Component):
    """
    Tile slots with one credit each, between one producer and one consumer process.

    With ``loader=True`` (the accumulator) a second producer, the loader, may fill a
    slot first and hand it to the producer (the ``LOADED`` state).

    Ports:
        Producer (the process that fills slots):
            prod_slot, prod_acquire: The head command's ``dst`` and ACQUIRE flag
            prod_resume: The head command's RESUME flag (``loader`` pools only)
            prod_credit: ``prod_slot`` is FREE (LOADED with ``prod_resume``)
            prod_take: Strobe: the head command was dispatched with ACQUIRE
            prod_signal, prod_signal_slot: Strobe: a command with SIGNAL completed
        Loader (``loader`` pools only; the BlockMover's accumulator LOAD):
            load_slot, load_acquire: The head command's ``dst`` and ACQUIRE flag
            load_credit: ``load_slot`` is FREE
            load_take: Strobe: the head command was dispatched with ACQUIRE
            load_signal, load_signal_slot: Strobe: a command with SIGNAL completed
                (OWNED -> LOADED)
        Consumer (the process that reads slots):
            cons_slot, cons_wait: The head command's ``src`` and WAIT flag
            cons_ready: ``cons_slot`` is READY
            cons_take: Strobe: the head command was dispatched with WAIT and RELEASE
            cons_release, cons_release_slot: Strobe: a command with RELEASE completed
        Status:
            free, ready: Per-slot state bits (OWNED, READING and LOADED = neither)
            error: Sticky protocol error (see the module docstring)
    """

    def __init__(self, slots: int, *, loader: bool = False):
        if not 1 <= slots <= 1 << SLOT_BITS:
            raise ValueError(f"a pool has 1..{1 << SLOT_BITS} slots, got {slots}")
        self.slots = slots
        self.loader = loader
        ports = {
            "prod_slot": In(SLOT_BITS),
            "prod_acquire": In(1),
            "prod_credit": Out(1),
            "prod_take": In(1),
            "prod_signal": In(1),
            "prod_signal_slot": In(SLOT_BITS),
            "cons_slot": In(SLOT_BITS),
            "cons_wait": In(1),
            "cons_ready": Out(1),
            "cons_take": In(1),
            "cons_release": In(1),
            "cons_release_slot": In(SLOT_BITS),
            "free": Out(slots),
            "ready": Out(slots),
            "error": Out(1),
        }
        if loader:
            ports.update(
                {
                    "prod_resume": In(1),
                    "load_slot": In(SLOT_BITS),
                    "load_acquire": In(1),
                    "load_credit": Out(1),
                    "load_take": In(1),
                    "load_signal": In(1),
                    "load_signal_slot": In(SLOT_BITS),
                }
            )
        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()
        n = self.slots
        state = [Signal(3 if self.loader else 2, name=f"slot{s}_state") for s in range(n)]
        resume = self.prod_resume if self.loader else Const(0)

        for s in range(n):
            m.d.comb += [
                self.free[s].eq(state[s] == SlotState.FREE),
                self.ready[s].eq(state[s] == SlotState.READY),
            ]
            with m.If(self.prod_slot == s):
                m.d.comb += self.prod_credit.eq(
                    state[s] == Mux(resume, SlotState.LOADED, SlotState.FREE)
                )
            with m.If(self.cons_slot == s):
                m.d.comb += self.cons_ready.eq(state[s] == SlotState.READY)
            if self.loader:
                with m.If(self.load_slot == s):
                    m.d.comb += self.load_credit.eq(state[s] == SlotState.FREE)

        error = Signal()
        m.d.comb += self.error.eq(error)

        def transition(strobe, slot, src, dst):
            sources = src if isinstance(src, tuple) else (src,)
            for s in range(n):
                with m.If(strobe & (slot == s)):
                    from_ok = state[s] == sources[0]
                    for other in sources[1:]:
                        from_ok = from_ok | (state[s] == other)
                    with m.If(from_ok):
                        m.d.sync += state[s].eq(dst)
                    with m.Else():
                        m.d.sync += error.eq(1)
            if n < 1 << SLOT_BITS:
                with m.If(strobe & (slot >= n)):
                    m.d.sync += error.eq(1)

        # Each transition starts from a different state, so at most one applies to a
        # slot in a cycle (a take and a RELEASE without WAIT of one READY slot at once
        # is a protocol error, and so are the two producers' takes or signals of one
        # slot at once)
        transition(self.prod_take & ~resume, self.prod_slot, SlotState.FREE, SlotState.OWNED)
        transition(self.prod_signal, self.prod_signal_slot, SlotState.OWNED, SlotState.READY)
        transition(self.cons_take, self.cons_slot, SlotState.READY, SlotState.READING)
        transition(
            self.cons_release, self.cons_release_slot,
            (SlotState.READING, SlotState.READY), SlotState.FREE,
        )  # fmt: skip
        with m.If(self.cons_take & self.cons_release & (self.cons_slot == self.cons_release_slot)):
            m.d.sync += error.eq(1)
        if self.loader:
            transition(self.prod_take & resume, self.prod_slot, SlotState.LOADED, SlotState.OWNED)
            transition(self.load_take, self.load_slot, SlotState.FREE, SlotState.OWNED)
            transition(self.load_signal, self.load_signal_slot, SlotState.OWNED, SlotState.LOADED)
            with m.If(
                (self.load_take & self.prod_take & (self.load_slot == self.prod_slot))
                | (
                    self.load_signal
                    & self.prod_signal
                    & (self.load_signal_slot == self.prod_signal_slot)
                )
            ):
                m.d.sync += error.eq(1)

        # A head command naming a slot outside the pool would wait forever
        if n < 1 << SLOT_BITS:
            outside = (self.prod_acquire & (self.prod_slot >= n)) | (
                self.cons_wait & (self.cons_slot >= n)
            )
            if self.loader:
                outside = outside | (self.load_acquire & (self.load_slot >= n))
            with m.If(outside):
                m.d.sync += error.eq(1)

        return m
