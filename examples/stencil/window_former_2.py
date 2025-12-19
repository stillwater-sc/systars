from amaranth.sim import Simulator, Tick

from systars.stencil import StencilConfig, WindowFormer

config = StencilConfig(
    max_width=8,
    max_height=8,
    max_kernel_h=3,
    max_kernel_w=3,
    parallel_channels=4,
)

wf = WindowFormer(config)


def testbench():
    kernel_w = 3
    yield wf.cfg_kernel_h.eq(3)
    yield wf.cfg_kernel_w.eq(kernel_w)
    yield wf.cfg_stride_w.eq(1)
    yield wf.cfg_width.eq(8)
    yield wf.out_ready.eq(1)
    yield Tick()

    # Check before input
    iv = yield wf.in_valid
    ir = yield wf.in_ready
    xfer = yield wf.dbg_input_xfer
    fc = yield wf.dbg_fill_count
    print(f"Before input: in_valid={iv}, in_ready={ir}, input_xfer={xfer}, fill_count={fc}")

    # Feed columns
    for col in range(5):
        yield wf.in_valid.eq(1)
        data = col | (col << 8) | (col << 16)
        yield wf.in_data.eq(data)

        # Check BEFORE tick
        iv = yield wf.in_valid
        ir = yield wf.in_ready
        xfer = yield wf.dbg_input_xfer
        print(f"Col {col} before tick: in_valid={iv}, in_ready={ir}, input_xfer={xfer}")

        yield Tick()

        fc = yield wf.dbg_fill_count
        cc = yield wf.dbg_col_counter
        wv = yield wf.window_valid
        print(f"Col {col} after tick: fill_count={fc}, col_counter={cc}, window_valid={wv}")


sim = Simulator(wf)
sim.add_clock(1e-6)
sim.add_testbench(testbench)
sim.run()
