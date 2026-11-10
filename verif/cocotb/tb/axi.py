"""
cocotb AXI4 memory model for ``SystolicTop``: L3 on the tile's AXI master.

``dram_model`` serves INCR bursts from a byte array and checks every burst it
accepts: beat aligned, inside one 4 KB page, and (for writes) WLAST on the last
beat. Writes honour the byte strobes, so a test can check that every byte outside
its outputs is untouched. Signals are found by prefix (``axi_arvalid``,
``axi_araddr``, ...), as ``SystolicTop`` names its AXI ports.

Port of kpugen's ``tb/axi.py`` ``dram_model``: kpugen reaches its tile's AXI master
through a memory port and watches it with ``burst_monitor``; ``SystolicTop`` exposes
the master itself, so the model serves it directly.

The model drives inputs right after a rising edge, samples in ReadOnly, and treats a
handshake as done at the next rising edge when valid and ready were both high in
ReadOnly.
"""

from __future__ import annotations

from cocotb.triggers import ReadOnly, RisingEdge
from systars.dma.axi import AXI_4KB


class _Port:
    """``port.name`` -> ``dut.<prefix><name>``."""

    def __init__(self, dut, prefix):
        self._dut = dut
        self._prefix = prefix

    def __getattr__(self, name):
        """The DUT signal ``<prefix><name>``."""
        return getattr(self._dut, f"{self._prefix}{name}")


async def dram_model(dut, memory: bytearray, beat: int, log: list, prefix="axi_") -> None:
    """Serve ``dut``'s AXI4 initiator ``<prefix>*`` from ``memory`` (runs forever).

    ``log`` receives ``(kind, address, beats)`` for every accepted burst.
    """
    p = _Port(dut, prefix)
    reads = []  # [address, beats left]
    write = None  # [address, beats done, beats]
    bresp_pending = False
    p.bresp.value = 0
    p.rresp.value = 0
    while True:
        p.arready.value = 1
        p.awready.value = int(write is None and not bresp_pending)
        p.wready.value = int(write is not None)
        p.bvalid.value = int(bresp_pending)
        p.rvalid.value = int(bool(reads))
        if reads:
            addr, left = reads[0]
            p.rdata.value = int.from_bytes(memory[addr : addr + beat], "little")
            p.rlast.value = int(left == 1)
        else:
            p.rlast.value = 0
        await ReadOnly()
        arvalid, rready = int(p.arvalid.value), int(p.rready.value)
        awvalid, wvalid, bready = int(p.awvalid.value), int(p.wvalid.value), int(p.bready.value)
        araddr = int(p.araddr.value) if arvalid else 0
        arlen = int(p.arlen.value) if arvalid else 0
        awaddr = int(p.awaddr.value) if awvalid else 0
        awlen = int(p.awlen.value) if awvalid else 0
        wdata = int(p.wdata.value) if wvalid else 0
        wstrb = int(p.wstrb.value) if wvalid else 0
        wlast = int(p.wlast.value) if wvalid else 0
        await RisingEdge(dut.clk)

        if reads and rready:
            reads[0][0] += beat
            reads[0][1] -= 1
            if reads[0][1] == 0:
                reads.pop(0)
        if arvalid:
            _check_burst(araddr, arlen + 1, beat)
            reads.append([araddr, arlen + 1])
            log.append(("r", araddr, arlen + 1))
        if bresp_pending and bready:
            bresp_pending = False
        if write is not None and wvalid:
            base = write[0] + write[1] * beat
            data = wdata.to_bytes(beat, "little")
            for i in range(beat):
                if wstrb >> i & 1:
                    memory[base + i] = data[i]
            write[1] += 1
            assert bool(wlast) == (write[1] == write[2]), (
                f"WLAST {wlast} on beat {write[1]} of {write[2]}"
            )
            if wlast:
                write, bresp_pending = None, True
        elif write is None and not bresp_pending and awvalid:
            _check_burst(awaddr, awlen + 1, beat)
            write = [awaddr, 0, awlen + 1]
            log.append(("w", awaddr, awlen + 1))


def _check_burst(addr: int, beats: int, beat: int) -> None:
    """Assert a burst is beat aligned and stays inside one 4 KB page."""
    assert addr % beat == 0, f"burst at {addr:#x} is not beat aligned"
    last = addr + beats * beat - 1
    assert addr // AXI_4KB == last // AXI_4KB, f"burst {addr:#x}+{beats} crosses 4 KB"
