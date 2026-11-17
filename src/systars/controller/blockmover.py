"""
BlockMover - the tile's one BlockMover: a CSP with two legs on one L3 port.

In the KPU's CSP model (kpu-sim, ``docs/01-architecture/kpu-execution-model.md``,
``BlockMover``) one process moves whole tiles between L3 and the tile's local memory,
in both directions. Here it has two legs, each with its own in-order command program
(its queue in ``SystolicTop``), because a LOAD and a STORE run at once on AXI's
independent read and write channels:

==========  ======================  =========================================================
Leg         Hop                     Fires when (``cmd_sync``, ``controller/sync.py``)
==========  ======================  =========================================================
inbound     L3 -> L2, or the        its destination slot's credit is free (ACQUIRE: an L2
(``bm_in``) Combiner (a LOAD)       slot, or a Combiner slot for a LOAD to a Combiner address)
                                    and the leg is idle. On completion it emits that slot's
                                    tile-ready token (SIGNAL).
outbound    the Combiner -> L3      the result tile's token is there (WAIT) and the leg is
(``bm_out``) (a STORE)              idle. It takes the token as it starts, and returns the
                                    Combiner slot's credit on completion (RELEASE).
==========  ======================  =========================================================

The L3 tile itself is taken as there: the host or the issuer of the commands orders a
LOAD after the DMA that fills L3 (there is no L3 token in the tile yet).

The legs share the BlockMover's one L3 port, the AXI master this component exposes:
the inbound leg drives its read channels, through ``StreamReader``, and the outbound
leg its write channels, through ``StreamWriter``. ``SystolicTop`` wires each leg's
command port and local-memory ports (``bm_in``, ``bm_out``) directly. The tile's
``BlockMoverPort`` (``noc/blockmover_port.py``) carries a global address's bursts over
the NoC.
"""

from typing import Any

from amaranth import Module
from amaranth.lib.wiring import Component, In, Out

from ..blockmover.reader import StreamReader
from ..blockmover.writer import StreamWriter
from ..config import SystolicConfig
from .blockmover_in import BlockMoverIn
from .blockmover_out import BlockMoverOut

# The L3 port's AXI channels, by direction from the BlockMover: (name, width, In/Out);
# a width is bits, or "bus" / "strb" (``axi_width``)
_READ: tuple[tuple[str, int | str, Any], ...] = (
    ("arvalid", 1, Out), ("arready", 1, In), ("araddr", 64, Out), ("arlen", 8, Out),
    ("arsize", 3, Out), ("arburst", 2, Out),
    ("rvalid", 1, In), ("rready", 1, Out), ("rdata", "bus", In), ("rlast", 1, In),
    ("rresp", 2, In),
)  # fmt: skip
_WRITE: tuple[tuple[str, int | str, Any], ...] = (
    ("awvalid", 1, Out), ("awready", 1, In), ("awaddr", 64, Out), ("awlen", 8, Out),
    ("awsize", 3, Out), ("awburst", 2, Out),
    ("wvalid", 1, Out), ("wready", 1, In), ("wdata", "bus", Out), ("wstrb", "strb", Out),
    ("wlast", 1, Out),
    ("bvalid", 1, In), ("bready", 1, Out), ("bresp", 2, In),
)  # fmt: skip
AXI_CHANNELS = _READ + _WRITE


def axi_width(width: int | str, config: SystolicConfig) -> int:
    """An AXI channel's width: bits, or ``"bus"`` (the data) or ``"strb"`` (its strobes)."""
    if width == "bus":
        return config.blockmover_buswidth
    if width == "strb":
        return config.blockmover_buswidth // 8
    assert isinstance(width, int)
    return width


class BlockMover(Component):
    """The BlockMover: its inbound and outbound legs and their engines (see the module
    docstring).

    Ports: ``axi_*``, the L3 port (AXI4 master, ``blockmover_buswidth`` data). The legs'
    command, status and local-memory ports are those of ``bm_in`` (``BlockMoverIn``) and
    ``bm_out`` (``BlockMoverOut``).
    """

    def __init__(self, config: SystolicConfig):
        self.config = config
        super().__init__({f"axi_{n}": d(axi_width(w, config)) for n, w, d in AXI_CHANNELS})
        # Built here, not in elaborate(), so kpu_gen.trace's probes reach their signals
        self.bm_in = BlockMoverIn(config)
        self.bm_out = BlockMoverOut(config)
        self.stream_reader = StreamReader(config)
        self.stream_writer = StreamWriter(config)

    def elaborate(self, _platform):
        m = Module()
        bm_in, bm_out = self.bm_in, self.bm_out
        m.submodules.bm_in = bm_in
        m.submodules.bm_out = bm_out
        m.submodules.stream_reader = reader = self.stream_reader
        m.submodules.stream_writer = writer = self.stream_writer

        # The inbound leg and its engine: L3 reads
        m.d.comb += [
            reader.req_valid.eq(bm_in.engine_req_valid),
            bm_in.engine_req_ready.eq(reader.req_ready),
            reader.req_addr.eq(bm_in.engine_req_addr),
            reader.req_len.eq(bm_in.engine_req_len),
            bm_in.engine_resp_valid.eq(reader.resp_valid),
            reader.resp_ready.eq(bm_in.engine_resp_ready),
            bm_in.engine_resp_data.eq(reader.resp_data),
            bm_in.engine_resp_last.eq(reader.resp_last),
        ]
        # The outbound leg and its engine: L3 writes
        m.d.comb += [
            writer.req_valid.eq(bm_out.engine_req_valid),
            bm_out.engine_req_ready.eq(writer.req_ready),
            writer.req_addr.eq(bm_out.engine_req_addr),
            writer.req_len.eq(bm_out.engine_req_len),
            writer.data_valid.eq(bm_out.engine_data_valid),
            bm_out.engine_data_ready.eq(writer.data_ready),
            writer.data.eq(bm_out.engine_data),
            writer.data_strb.eq(bm_out.engine_data_strb),
            writer.data_last.eq(bm_out.engine_data_last),
            bm_out.engine_done.eq(writer.done),
        ]
        # The L3 port: the read channels are the reader's, the write channels the writer's
        for engine, channels in ((reader, _READ), (writer, _WRITE)):
            for name, _, direction in channels:
                ours, theirs = getattr(self, f"axi_{name}"), getattr(engine, f"mem_{name}")
                m.d.comb += ours.eq(theirs) if direction is Out else theirs.eq(ours)
        return m
