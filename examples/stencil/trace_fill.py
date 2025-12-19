# Let's trace through the logic step by step manually
from amaranth import Module, Signal
from amaranth.lib.wiring import Component, In, Out
from amaranth.sim import Simulator, Tick


# Create a minimal test to understand fill_count behavior
class MinimalWindowFormer(Component):
    def __init__(self):
        super().__init__(
            {
                "in_valid": In(1),
                "in_ready": Out(1),
                "out_ready": In(1),
                "cfg_kernel_w": In(4),
                "window_valid": Out(1),
                "fill_count_out": Out(4),  # Expose for debugging
            }
        )

    def elaborate(self, _platform):
        m = Module()

        fill_count = Signal(4, name="fill_count")
        m.d.comb += self.fill_count_out.eq(fill_count)

        window_is_valid = Signal(name="window_is_valid")
        m.d.comb += window_is_valid.eq(fill_count >= self.cfg_kernel_w)
        m.d.comb += self.window_valid.eq(window_is_valid)

        with m.If(window_is_valid):
            m.d.comb += self.in_ready.eq(self.out_ready)
        with m.Else():
            m.d.comb += self.in_ready.eq(1)

        input_xfer = Signal(name="input_xfer")
        m.d.comb += input_xfer.eq(self.in_valid & self.in_ready)

        # Increment fill_count on input transfer
        with m.If(input_xfer), m.If(fill_count < 7):  # Max value
            m.d.sync += fill_count.eq(fill_count + 1)

        return m


wf = MinimalWindowFormer()


def testbench():
    yield wf.cfg_kernel_w.eq(3)
    yield wf.out_ready.eq(1)
    yield Tick()

    # Check initial state
    fc = yield wf.fill_count_out
    wv = yield wf.window_valid
    ir = yield wf.in_ready
    print(f"Initial: fill_count={fc}, window_valid={wv}, in_ready={ir}")

    for col in range(6):
        yield wf.in_valid.eq(1)
        yield Tick()

        fc = yield wf.fill_count_out
        wv = yield wf.window_valid
        ir = yield wf.in_ready
        print(f"After col {col}: fill_count={fc}, window_valid={wv}, in_ready={ir}")


sim = Simulator(wf)
sim.add_clock(1e-6)
sim.add_testbench(testbench)
sim.run()
