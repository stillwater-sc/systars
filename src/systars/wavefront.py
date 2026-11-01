"""
Wavefront tracing of the systolic array RTL.

Runs the real ``SystolicArray`` in Amaranth simulation, records what every PE
sees in every cycle, and checks it against the wavefront the design promises:

- Output-stationary: A[i, k] and B[k, j] meet at PE(i, j) in cycle
  ``k + ti + tj`` (``ti = i // tile_rows``, ``tj = j // tile_cols``, the PEArray
  row and column), with ``valid``. Inside a PEArray data passes combinationally,
  so the wavefront advances one PEArray per cycle, not one PE.
- Weight-stationary: row m of A meets W[r, j] at PE(r, j) in cycle
  ``m + ti + tj``, and the partial sum arriving there is
  ``A[m, :r] @ W[:r, j]``.

``label_operands`` builds operands whose values say which element they are,
so the recorded frames can show, from the RTL's own values, which ``k`` (OS)
or ``m`` (WS) each PE is working on, and flag a PE whose A and B disagree.
``render_text`` and ``render_svg`` draw the frames, with the PEArray boundaries
(where the pipeline registers sit) marked.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field

import numpy as np
from amaranth.sim import Simulator

from .config import SystolicConfig
from .core.pe import OUTPUT_STATIONARY, WEIGHT_STATIONARY
from .core.systolic_array import SystolicArray

OS, WS = OUTPUT_STATIONARY, WEIGHT_STATIONARY


@dataclass
class Frame:
    """What every PE sees in one cycle (arrays are rows x cols)."""

    cycle: int
    phase: str  # "preload" (WS weights), "stream", or "drain" (OS results)
    valid: np.ndarray
    shift: np.ndarray
    a: np.ndarray
    b: np.ndarray
    psum_in: np.ndarray
    psum_out: np.ndarray
    chain: np.ndarray  # the shift-chain register: c (OS) or w (WS)


@dataclass
class Trace:
    """A recorded run. ``A``/``B`` are the operands (WS: ``B`` is the weights W)."""

    config: SystolicConfig
    dataflow: int
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray  # the result as the RTL produced it
    frames: list[Frame] = field(default_factory=list)
    latency: int = 0  # SystolicArray.latency
    last_cycles: list[int] = field(default_factory=list)  # stream cycles with out_last high

    @property
    def rows(self) -> int:
        return self.config.grid_rows * self.config.tile_rows

    @property
    def cols(self) -> int:
        return self.config.grid_cols * self.config.tile_cols

    def offset(self, i: int, j: int) -> int:
        """Pipeline registers between the array inputs and PE(i, j)."""
        return i // self.config.tile_rows + j // self.config.tile_cols

    def expected_index(self, frame: Frame, i: int, j: int) -> int | None:
        """The k (OS) or m (WS) that should be at PE(i, j) in ``frame``, if any."""
        if frame.phase != "stream":
            return None
        n = self.A.shape[1] if self.dataflow == OS else self.A.shape[0]
        index = frame.cycle - self.offset(i, j)
        return index if 0 <= index < n else None

    def check(self) -> list[str]:
        """Every place the RTL departs from the predicted wavefront (empty: correct).

        Checks every PE in every stream cycle, when ``out_last`` fired, and the result.
        Preload and drain frames are recorded for drawing but not checked cycle by cycle.
        """
        problems = []
        A, B = self.A, self.B
        n = A.shape[1] if self.dataflow == OS else A.shape[0]
        expected_last = [n - 1 + self.latency] if n else []
        if self.last_cycles != expected_last:
            problems.append(
                f"out_last in stream cycles {self.last_cycles}, expected {expected_last}"
            )
        for f in self.frames:
            if f.phase != "stream":
                continue
            for i in range(self.rows):
                for j in range(self.cols):
                    idx = self.expected_index(f, i, j)
                    where = f"cycle {f.cycle}, PE({i},{j})"
                    if bool(f.valid[i, j]) != (idx is not None):
                        problems.append(
                            f"{where}: valid={int(f.valid[i, j])}, expected {int(idx is not None)}"
                        )
                        continue
                    if idx is None:
                        continue
                    if self.dataflow == OS:
                        if (f.a[i, j], f.b[i, j]) != (A[i, idx], B[idx, j]):
                            problems.append(
                                f"{where}: a={f.a[i, j]} b={f.b[i, j]}, expected "
                                f"A[{i},{idx}]={A[i, idx]} B[{idx},{j}]={B[idx, j]}"
                            )
                    else:
                        psum = int(A[idx, :i] @ B[:i, j])
                        if (
                            f.a[i, j] != A[idx, i]
                            or f.psum_in[i, j] != psum
                            or f.chain[i, j] != B[i, j]
                        ):
                            problems.append(
                                f"{where}: a={f.a[i, j]} psum_in={f.psum_in[i, j]} w={f.chain[i, j]}, "
                                f"expected A[{idx},{i}]={A[idx, i]} psum={psum} W[{i},{j}]={B[i, j]}"
                            )
        if not np.array_equal(self.C, A @ B):
            problems.append("result differs from A @ B")
        return problems

    def decode(self, frame: Frame, i: int, j: int) -> tuple[int | None, bool]:
        """The k (OS) or m (WS) the RTL values at PE(i, j) belong to, and whether A and B agree.

        Needs operands from ``label_operands``; returns (None, True) for an idle PE.
        """
        if frame.phase != "stream" or not frame.valid[i, j]:
            return None, True
        if self.dataflow == OS:
            k_dim = self.A.shape[1]
            k_a = int(frame.a[i, j]) - 1 - i * k_dim
            k_b = -int(frame.b[i, j]) - 1 - j * k_dim
            return k_a, k_a == k_b
        m = (int(frame.a[i, j]) - 1 - i) // self.rows
        return m, True


def label_operands(config: SystolicConfig, dataflow: int, n: int, *, seed: int = 0):
    """Operands whose values identify their elements (int8).

    OS: A is rows x n with A[i, k] = 1 + i*n + k, and B is n x cols with
    B[k, j] = -(1 + j*n + k), so ``k`` can be read back from either.
    WS: A is n x rows with A[m, r] = 1 + m*rows + r; W is random.
    """
    rows = config.grid_rows * config.tile_rows
    cols = config.grid_cols * config.tile_cols
    rng = np.random.default_rng(seed)
    if dataflow == OS:
        if rows * n > 127 or cols * n > 128:
            raise ValueError("labelled OS operands need rows * n <= 127 and cols * n <= 128")
        A = np.array([[1 + i * n + k for k in range(n)] for i in range(rows)], dtype=np.int64)
        B = np.array([[-(1 + j * n + k) for j in range(cols)] for k in range(n)], dtype=np.int64)
        return A.reshape(rows, n), B.reshape(n, cols)
    if n * rows > 127:
        raise ValueError("labelled WS operands need n * rows <= 127")
    A = np.array([[1 + m * rows + r for r in range(rows)] for m in range(n)], dtype=np.int64)
    W = rng.integers(-8, 8, (rows, cols))
    return A.reshape(n, rows), W


def record(config: SystolicConfig, dataflow: int, A, B) -> Trace:
    """Run ``A @ B`` on the RTL array and record every PE in every cycle.

    OS: A is rows x K, B is K x cols. WS: A is M x rows, B is the rows x cols weights.
    Stream cycles are numbered from 0 (the first A vector); preload cycles are
    negative.
    """
    A, B = np.asarray(A), np.asarray(B)
    dut = SystolicArray(config)
    rows = config.grid_rows * config.tile_rows
    cols = config.grid_cols * config.tile_cols
    trace = Trace(config, dataflow, A, B, np.zeros((0, cols), dtype=np.int64), latency=dut.latency)
    sim = Simulator(dut)  # elaborates the design, so the PE references below exist
    pes = [
        [
            dut.pe_arrays[i // config.tile_rows][j // config.tile_cols].pes[i % config.tile_rows][
                j % config.tile_cols
            ]
            for j in range(cols)
        ]
        for i in range(rows)
    ]

    def snapshot(ctx, cycle, phase):
        def grid(name):
            return np.array(
                [[ctx.get(getattr(pes[i][j], name)) for j in range(cols)] for i in range(rows)]
            )

        trace.frames.append(
            Frame(
                cycle,
                phase,
                grid("in_valid").astype(bool),
                grid("in_shift").astype(bool),
                grid("in_a"),
                grid("in_b"),
                grid("in_psum"),
                grid("out_psum"),
                grid("out_c"),
            )  # fmt: skip
        )

    def drive(ctx, **values):
        for name, value in values.items():
            ctx.set(getattr(dut, name), int(value))

    async def run_os(ctx):
        k_dim = A.shape[1]
        drive(ctx, in_control_dataflow=OS)
        # The last product reaches the bottom-right PEArray in cycle k_dim - 1 + latency.
        # A fixed range (not "until out_last") keeps a control-path bug from hanging
        # the trace; check() then reports the missing out_last.
        for cycle in range(k_dim + dut.latency):
            streaming = cycle < k_dim
            drive(ctx, in_valid=streaming, in_last=cycle == k_dim - 1)
            for i in range(rows):
                drive(ctx, **{f"in_a_{i}": A[i, cycle] if streaming else 0})
            for j in range(cols):
                drive(ctx, **{f"in_b_{j}": B[cycle, j] if streaming else 0})
            snapshot(ctx, cycle, "stream")
            if ctx.get(dut.out_last):
                trace.last_cycles.append(cycle)
            await ctx.tick()
        cycle = k_dim + dut.latency
        drive(ctx, in_valid=0, in_last=0, in_shift=1)
        drained = []
        for _ in range(rows):
            snapshot(ctx, cycle, "drain")
            drained.append([ctx.get(getattr(dut, f"out_c_{j}")) for j in range(cols)])
            await ctx.tick()
            cycle += 1
        drive(ctx, in_shift=0)
        trace.C = np.array(drained[::-1])

    async def run_ws(ctx):
        m_rows = A.shape[0]
        drive(ctx, in_control_dataflow=WS, in_shift=1)
        for n, r in enumerate(reversed(range(rows))):
            for j in range(cols):
                drive(ctx, **{f"in_d_{j}": B[r, j]})
            snapshot(ctx, n - rows, "preload")
            await ctx.tick()
        drive(ctx, in_shift=0)
        out = []
        for cycle in range(m_rows + dut.latency + 1):
            streaming = cycle < m_rows
            drive(ctx, in_valid=streaming, in_last=cycle == m_rows - 1)
            for i in range(rows):
                drive(ctx, **{f"in_a_{i}": A[cycle, i] if streaming else 0})
            snapshot(ctx, cycle, "stream")
            if ctx.get(dut.out_last):
                trace.last_cycles.append(cycle)
            if ctx.get(dut.out_valid):
                out.append([ctx.get(getattr(dut, f"out_psum_{j}")) for j in range(cols)])
            await ctx.tick()
        trace.C = np.array(out).reshape(-1, cols)

    sim.add_clock(1e-6)
    sim.add_testbench(run_os if dataflow == OS else run_ws)
    sim.run()
    return trace


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell_text(trace: Trace, frame: Frame, i: int, j: int) -> tuple[str, str]:
    """(label, kind) for one PE in one frame; kind is idle/active/mismatch/shift."""
    if frame.phase == "preload":
        return f"w{frame.chain[i, j]}", "shift"
    if frame.phase == "drain":
        return str(frame.chain[i, j]), "shift"
    index, agree = trace.decode(frame, i, j)
    if index is None:
        return ".", "idle"
    if not agree:
        return "X", "mismatch"
    return ("k" if trace.dataflow == OS else "m") + str(index), "active"


def render_text(trace: Trace, width: int = 7) -> str:
    """The frames as text, one grid per cycle; '|' and '-' mark PEArray boundaries."""
    cfg = trace.config
    out = []
    for f in trace.frames:
        out.append(f"cycle {f.cycle:>3}  {f.phase}")
        for i in range(trace.rows):
            if i and i % cfg.tile_rows == 0:
                out.append("    " + "-" * ((width + 1) * trace.cols + cfg.grid_cols - 1))
            cells = []
            for j in range(trace.cols):
                if j and j % cfg.tile_cols == 0:
                    cells.append("|")
                cells.append(_cell_text(trace, f, i, j)[0].center(width))
            out.append("    " + " ".join(cells))
        out.append("")
    return "\n".join(out)


_FILL = {"idle": "#f2f2f2", "mismatch": "#d62728", "shift": "#c7d7ef"}


def _active_fill(index: int, n: int) -> str:
    """A distinct colour per k (or m), so a wavefront is one colour.

    Hues run from green to violet, keeping clear of the red used for a mismatch.
    """
    hue = 100 + 190 * index / max(n - 1, 1)
    return f"hsl({hue:.0f}, 60%, 66%)"


def render_svg(trace: Trace, *, cell: int = 56, seconds_per_frame: float = 0.8) -> str:
    """An animated SVG (SMIL) of the frames, looping, viewable in a browser."""
    cfg = trace.config
    rows, cols = trace.rows, trace.cols
    margin, header, char_w = 24, 56, 7.6  # char_w: monospace advance at 12-14px
    legend = [
        f"{'kN' if trace.dataflow == OS else 'mN'}: the element meeting at that PE "
        "(one colour per wavefront)",
        "thick lines: PEArray boundaries, where the pipeline registers sit",
    ]
    n_frames = len(trace.frames)
    total = n_frames * seconds_per_frame
    n_index = trace.A.shape[1] if trace.dataflow == OS else trace.A.shape[0]
    mode = "output-stationary" if trace.dataflow == OS else "weight-stationary"
    title = (
        f"{mode}: {cfg.grid_rows}x{cfg.grid_cols} grid of {cfg.tile_rows}x{cfg.tile_cols} PEArrays"
    )
    text_w = max(len(s) for s in [title, *legend]) * char_w
    width = int(2 * margin + max(cols * cell, text_w))
    height = header + rows * cell + 2 * margin + 18

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{margin}" y="20" font-size="14">{html.escape(title)}</text>',
    ]
    for n, f in enumerate(trace.frames):
        # Visible only during this frame's slot of the loop (discrete timing)
        start, end = n / n_frames, (n + 1) / n_frames
        if n == 0:
            key_times, values = f"0;{end:.6f}", "visible;hidden"
        else:
            key_times, values = f"0;{start:.6f};{end:.6f}", "hidden;visible;hidden"
        parts.append(
            # Frame 0 is visible without animation support (a static viewer shows it)
            f'<g visibility="{"visible" if n == 0 else "hidden"}">'
            f'<animate attributeName="visibility" calcMode="discrete" '
            f'dur="{total:.2f}s" repeatCount="indefinite" keyTimes="{key_times}" values="{values}"/>'
        )
        parts.append(
            f'<text x="{margin}" y="42" font-size="13">cycle {f.cycle}  {f.phase}'
            f"{'  (shift)' if f.phase != 'stream' else ''}</text>"
        )
        for i in range(rows):
            for j in range(cols):
                label, kind = _cell_text(trace, f, i, j)
                fill = _active_fill(int(label[1:]), n_index) if kind == "active" else _FILL[kind]
                x, y = margin + j * cell, header + i * cell
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" '
                    f'stroke="#999" stroke-width="0.5"/>'
                    f'<text x="{x + cell / 2}" y="{y + cell / 2 + 4}" font-size="12" '
                    f'text-anchor="middle">{html.escape(label)}</text>'
                )
        parts.append("</g>")
    # PEArray boundaries: where the pipeline registers sit
    for tr in range(cfg.grid_rows + 1):
        y = header + tr * cfg.tile_rows * cell
        parts.append(
            f'<line x1="{margin}" y1="{y}" x2="{margin + cols * cell}" y2="{y}" stroke="black" stroke-width="2"/>'
        )
    for tc in range(cfg.grid_cols + 1):
        x = margin + tc * cfg.tile_cols * cell
        parts.append(
            f'<line x1="{x}" y1="{header}" x2="{x}" y2="{header + rows * cell}" stroke="black" stroke-width="2"/>'
        )
    for line, text in enumerate(legend):
        y = header + rows * cell + margin + line * 16
        parts.append(f'<text x="{margin}" y="{y}" font-size="12">{html.escape(text)}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
