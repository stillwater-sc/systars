"""
A byte-addressed AXI memory model for SystolicTop tests, standing in for L3.

Reads return the beat at the (beat-aligned) burst address; writes honour the byte
strobes, so a test can check that a partial-row STORE leaves the bytes around it
untouched. Bursts must start on a beat boundary (asserted), as the tile's
BlockMover issues them.

``stall`` (0..1) is L3 backpressure: on that fraction of cycles, chosen by a
seeded generator, the model holds back every channel (no address accepted, no
read beat offered, no write beat accepted).
"""

from __future__ import annotations

import random


def axi_memory(dut, memory: bytearray, beat: int, *, stall: float = 0.0, seed: int = 0):
    """An Amaranth simulator process serving ``dut``'s AXI ports from ``memory``."""
    rng = random.Random(seed)

    def beat_at(addr):
        return int.from_bytes(memory[addr : addr + beat], "little")

    async def process(ctx):
        reads = []  # [address, beats left]
        write = None  # [address, beat index]
        bresp_pending = False
        while True:
            go = not (stall and rng.random() < stall)
            ctx.set(dut.axi_arready, go)
            ctx.set(dut.axi_awready, go and write is None and not bresp_pending)
            ctx.set(dut.axi_wready, go and write is not None)
            ctx.set(dut.axi_bvalid, bresp_pending)
            if reads and go:
                addr, left = reads[0]
                ctx.set(dut.axi_rvalid, 1)
                ctx.set(dut.axi_rdata, beat_at(addr))
                ctx.set(dut.axi_rlast, left == 1)
            else:
                ctx.set(dut.axi_rvalid, 0)
                ctx.set(dut.axi_rlast, 0)
            (
                _,
                _,
                arvalid,
                araddr,
                arlen,
                rready,
                awvalid,
                awaddr,
                wvalid,
                wdata,
                wstrb,
                wlast,
                bready,
            ) = await ctx.tick().sample(
                dut.axi_arvalid,
                dut.axi_araddr,
                dut.axi_arlen,
                dut.axi_rready,
                dut.axi_awvalid,
                dut.axi_awaddr,
                dut.axi_wvalid,
                dut.axi_wdata,
                dut.axi_wstrb,
                dut.axi_wlast,
                dut.axi_bready,
            )
            if reads and go and rready:
                reads[0][0] += beat
                reads[0][1] -= 1
                if reads[0][1] == 0:
                    reads.pop(0)
            if go and arvalid:
                assert araddr % beat == 0, f"read burst at {araddr:#x} is not beat-aligned"
                reads.append([araddr, arlen + 1])
            if bresp_pending and bready:
                bresp_pending = False
            if not go:
                pass
            elif write is not None and wvalid:
                base = write[0] + write[1] * beat
                data = wdata.to_bytes(beat, "little")
                for i in range(beat):
                    if wstrb >> i & 1:
                        memory[base + i] = data[i]
                write[1] += 1
                if wlast:
                    write, bresp_pending = None, True
            elif write is None and not bresp_pending and awvalid:
                assert awaddr % beat == 0, f"write burst at {awaddr:#x} is not beat-aligned"
                write = [awaddr, 0]

    return process
