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

    cfg_kw = yield wf.cfg_kernel_w
    fc = yield wf.dbg_fill_count
    wv = yield wf.window_valid
    ir = yield wf.in_ready
    print(f"Initial: cfg_kernel_w={cfg_kw}, fill_count={fc}, window_valid={wv}, in_ready={ir}")

    # Feed columns
    for col in range(5):
        yield wf.in_valid.eq(1)
        data = col | (col << 8) | (col << 16)
        yield wf.in_data.eq(data)
        yield Tick()

        fc = yield wf.dbg_fill_count
        wv = yield wf.window_valid
        ir = yield wf.in_ready
        print(f"After col {col}: fill_count={fc}, window_valid={wv}, in_ready={ir}")


sim = Simulator(wf)
sim.add_clock(1e-6)
sim.add_testbench(testbench)
sim.run()
