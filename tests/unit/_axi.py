"""
A byte-addressed AXI memory model for SystolicTop tests, standing in for L3.

Reads return the beat at the (beat-aligned) burst address; writes honour the byte
strobes, so a test can check that a partial-row STORE leaves the bytes around it
untouched. Bursts must start on a beat boundary (asserted), as the tile's
BlockMover issues them.
"""

from __future__ import annotations


def axi_memory(dut, memory: bytearray, beat: int):
    """An Amaranth simulator process serving ``dut``'s AXI ports from ``memory``."""

    def beat_at(addr):
        return int.from_bytes(memory[addr : addr + beat], "little")

    async def process(ctx):
        reads = []  # [address, beats left]
        write = None  # [address, beat index]
        bresp_pending = False
        while True:
            ctx.set(dut.axi_arready, 1)
            ctx.set(dut.axi_awready, write is None and not bresp_pending)
            ctx.set(dut.axi_wready, write is not None)
            ctx.set(dut.axi_bvalid, bresp_pending)
            if reads:
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
            if reads and rready:
                reads[0][0] += beat
                reads[0][1] -= 1
                if reads[0][1] == 0:
                    reads.pop(0)
            if arvalid:
                assert araddr % beat == 0, f"read burst at {araddr:#x} is not beat-aligned"
                reads.append([araddr, arlen + 1])
            if bresp_pending and bready:
                bresp_pending = False
            if write is not None and wvalid:
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
