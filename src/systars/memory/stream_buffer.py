"""
L1: the stream buffers that feed the fabric and the result buffers that drain it.

**StreamBuffers** are ``count`` stream buffers, one per array row (A, flowing east)
or per array column (B, flowing south). Each holds one block: an L2 row of up to
``block`` elements. The Streamer reads one L2 row and writes it, in parallel, into
one buffer (``load_*``; one load per clock); the buffer then streams it out one
element per clock, element 0 first, starting the clock after the load.

- One block of storage is one credit. ``credit_i`` is high when buffer ``i`` can
  take a block this clock: it is empty, or its last element leaves this clock.
  So a buffer loaded in the clock its last element leaves streams the next block
  with no gap, and tiles follow each other with no bubbles.
- A load into a buffer without its credit is a protocol error (``error``, sticky):
  it would overwrite elements still to be streamed. The Streamer's schedule never
  does it.
- Each block carries its stream length (``load_len``, at most ``block``) and two
  markers: ``first`` goes out with its element 0 and ``last`` with its final
  element (an OS reduction's first and last products; a reduction can span
  several blocks).

**ResultBuffers** are ``lanes`` result buffers, one per array column, at the edge
the results leave (north for OS, south for the stationary mode). Each is a
first-in, first-out buffer of up to ``depth`` results. The fabric
delivers a column's results in row order, so once every column holds its next
result, those results form one row of the result tile: the buffers emit it
(``out_valid``, ``out_data``, column 0 in the low bits) and drop it. That turns the
columns back into the row-major rows the accumulator and STORE expect.

- ``credit_j`` is high while buffer ``j`` has room; a result arriving at a full
  buffer is a protocol error (``error``, sticky).
- OS results reach the north edge every other clock, so the buffers emit a row
  every other clock; stationary results arrive every clock, and so do the rows.
"""

from amaranth import Cat, Module, Mux, Signal, signed
from amaranth.lib.fifo import SyncFIFO
from amaranth.lib.wiring import Component, In, Out


class StreamBuffers(Component):
    """
    ``count`` stream buffers of one ``block``-element block of ``element_bits``-bit elements.

    Ports:
        load_valid, load_index, load_data, load_len, load_first, load_last:
            Write one block into buffer ``load_index`` this clock
        out_i, out_valid_i, out_first_i, out_last_i: Buffer ``i``'s stream
        credit_i: Buffer ``i`` can take a block this clock
        busy: Some buffer is still streaming
        error: Sticky: a block was written into a buffer without its credit
    """

    def __init__(self, count: int, block: int, element_bits: int):
        self.count, self.block, self.element_bits = count, block, element_bits
        ports = {
            "load_valid": In(1),
            "load_index": In(range(count)),
            "load_data": In(block * element_bits),
            "load_len": In(range(block + 1)),
            "load_first": In(1),
            "load_last": In(1),
            "busy": Out(1),
            "error": Out(1),
        }
        for i in range(count):
            ports[f"out_{i}"] = Out(signed(element_bits))
            ports[f"out_valid_{i}"] = Out(1)
            ports[f"out_first_{i}"] = Out(1)
            ports[f"out_last_{i}"] = Out(1)
            ports[f"credit_{i}"] = Out(1)
        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()
        block, eb = self.block, self.element_bits

        error = Signal()
        m.d.comb += self.error.eq(error)
        busy = []
        for i in range(self.count):
            data = Signal(block * eb, name=f"data_{i}")
            left = Signal(range(block + 1), name=f"left_{i}")  # elements still to stream
            sent = Signal(range(block + 1), name=f"sent_{i}")  # elements streamed so far
            first = Signal(name=f"first_{i}")
            last = Signal(name=f"last_{i}")
            streaming = left != 0
            credit = left <= 1
            busy.append(streaming)

            m.d.comb += [
                getattr(self, f"out_{i}").eq(data[:eb].as_signed()),
                getattr(self, f"out_valid_{i}").eq(streaming),
                getattr(self, f"out_first_{i}").eq(streaming & first & (sent == 0)),
                getattr(self, f"out_last_{i}").eq(streaming & last & (left == 1)),
                getattr(self, f"credit_{i}").eq(credit),
            ]
            with m.If(self.load_valid & (self.load_index == i)):
                m.d.sync += [
                    data.eq(self.load_data),
                    left.eq(Mux(self.load_len > block, block, self.load_len)),
                    sent.eq(0),
                    first.eq(self.load_first),
                    last.eq(self.load_last),
                ]
                with m.If(~credit):
                    m.d.sync += error.eq(1)
            with m.Elif(streaming):
                m.d.sync += [data.eq(data >> eb), left.eq(left - 1), sent.eq(sent + 1)]

        m.d.comb += self.busy.eq(Cat(*busy).any())
        return m


class ResultBuffers(Component):
    """
    ``lanes`` column result buffers of ``depth`` ``element_bits``-bit results, emitting rows.

    Ports:
        in_j, in_valid_j: A result from column ``j`` this clock
        out_valid, out_data: A row of results (every buffer's oldest), this clock
        credit_j: Buffer ``j`` has room for a result
        busy: Some buffer holds results
        error: Sticky: a result arrived at a full buffer
    """

    def __init__(self, lanes: int, depth: int, element_bits: int):
        self.lanes, self.depth, self.element_bits = lanes, depth, element_bits
        ports = {
            "out_valid": Out(1),
            "out_data": Out(lanes * element_bits),
            "busy": Out(1),
            "error": Out(1),
        }
        for j in range(lanes):
            ports[f"in_{j}"] = In(signed(element_bits))
            ports[f"in_valid_{j}"] = In(1)
            ports[f"credit_{j}"] = Out(1)
        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()
        lanes, eb = self.lanes, self.element_bits

        fifos = [SyncFIFO(width=eb, depth=self.depth) for _ in range(lanes)]
        error = Signal()
        m.d.comb += self.error.eq(error)

        row_ready = Signal()
        m.d.comb += row_ready.eq(Cat(*[f.r_rdy for f in fifos]).all())

        for j, fifo in enumerate(fifos):
            m.submodules[f"column_{j}"] = fifo
            m.d.comb += [
                fifo.w_data.eq(getattr(self, f"in_{j}")),
                fifo.w_en.eq(getattr(self, f"in_valid_{j}")),
                fifo.r_en.eq(row_ready),
                getattr(self, f"credit_{j}").eq(fifo.w_rdy),
            ]
            with m.If(getattr(self, f"in_valid_{j}") & ~fifo.w_rdy):
                m.d.sync += error.eq(1)

        m.d.comb += [
            self.out_valid.eq(row_ready),
            self.out_data.eq(Cat(*[f.r_data for f in fifos])),
            self.busy.eq(Cat(*[f.r_rdy for f in fifos]).any()),
        ]
        return m
