"""
Wavefront tracing of the systolic array RTL.

Runs the real ``SystolicArray`` in Amaranth simulation, driven the way the L1
stream buffers drive it (row ``i``'s stream starts one clock after row ``i - 1``'s,
column ``j``'s one after column ``j - 1``'s), records what every PE holds in every
cycle, and checks it against the wavefront the design promises. Every PE registers
its inputs, so the wavefront advances one PE per cycle in every direction:

- Output-stationary: A[i, k] and B[k, j] meet at PE(i, j) in cycle
  ``1 + i + j + k``, with ``valid``. After the last product the results move north,
  one PE per cycle: PE(i, j)'s result leaves the north edge of column ``j`` in cycle
  ``K + j + 2 i`` (``SystolicArray.os_result_clock``).
- Stationary (WS; IS is the same fabric mode): element ``m`` of the stream meets
  W[r, j] at PE(r, j) in cycle ``1 + r + j + m``, and the partial sum it leaves with
  is ``A[m, :r+1] @ W[:r+1, j]``.

``label_operands`` builds operands whose values say which element they are,
so the recorded frames can show, from the RTL's own values, which ``k`` (OS)
or ``m`` (WS) each PE is working on, and flag a PE whose A and B disagree.
``render_text`` and ``render_svg`` draw the frames, with the PEArray boundaries
(structural only) marked.
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


def stream_start(lane: int) -> int:
    """The clock lane ``lane``'s stream starts (the L1 stream buffers' stagger)."""
    return lane


@dataclass
class Frame:
    """What every PE holds in one cycle (arrays are rows x cols)."""

    cycle: int
    phase: str  # "preload" (WS weights), "stream", or "drain" (OS results moving north)
    valid: np.ndarray
    shift: np.ndarray
    a: np.ndarray
    b: np.ndarray
    psum_out: np.ndarray
    w: np.ndarray  # the stationary register
    c: np.ndarray  # the OS result each PE sends north this cycle
    c_valid: np.ndarray


@dataclass
class Trace:
    """A recorded run. ``A``/``B`` are the operands (WS: ``B`` is the weights W)."""

    config: SystolicConfig
    dataflow: int
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray  # the result as the RTL produced it
    frames: list[Frame] = field(default_factory=list)
    result_clocks: dict = field(default_factory=dict)  # (row, col) -> cycle (OS)

    @property
    def rows(self) -> int:
        return self.config.grid_rows * self.config.tile_rows

    @property
    def cols(self) -> int:
        return self.config.grid_cols * self.config.tile_cols

    @property
    def length(self) -> int:
        """K (OS) or M (WS): the stream length."""
        return self.A.shape[1] if self.dataflow == OS else self.A.shape[0]

    def expected_index(self, frame: Frame, i: int, j: int) -> int | None:
        """The k (OS) or m (WS) that should be at PE(i, j) in ``frame``, if any."""
        if frame.phase != "stream":
            return None
        index = frame.cycle - 1 - i - j
        return index if 0 <= index < self.length else None

    def check(self) -> list[str]:
        """Every place the RTL departs from the predicted wavefront (empty: correct).

        Checks every PE in every stream cycle, the clock each OS result leaves the
        north edge, and the result. Preload frames are recorded for drawing only.
        """
        problems = []
        A, B = self.A, self.B
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
                        psum = int(A[idx, : i + 1] @ B[: i + 1, j])
                        if (
                            f.a[i, j] != A[idx, i]
                            or f.psum_out[i, j] != psum
                            or f.w[i, j] != B[i, j]
                        ):
                            problems.append(
                                f"{where}: a={f.a[i, j]} psum={f.psum_out[i, j]} w={f.w[i, j]}, "
                                f"expected A[{idx},{i}]={A[idx, i]} psum={psum} W[{i},{j}]={B[i, j]}"
                            )
        if self.dataflow == OS and self.length:
            expected = {
                (i, j): SystolicArray.os_result_clock(0, self.length, i, j)
                for i in range(self.rows)
                for j in range(self.cols)
            }
            if self.result_clocks != expected:
                missing = len(expected) - len(self.result_clocks)
                problems.append(
                    f"results left the north edge at the wrong clocks ({missing} missing)"
                )
        expected_C = A @ B if self.length else np.zeros((0, self.cols), dtype=np.int64)
        if self.C.shape != expected_C.shape or not np.array_equal(self.C, expected_C):
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

    OS: A is rows x K, B is K x cols, one reduction. WS: A is M x rows, B is the
    rows x cols weights. Stream cycles are numbered from 0 (row 0's and column 0's
    first element on the edge); preload cycles are negative.
    """
    A, B = np.asarray(A), np.asarray(B)
    dut = SystolicArray(config)
    rows, cols = dut.rows, dut.cols
    trace = Trace(config, dataflow, A, B, np.zeros((0, cols), dtype=np.int64))
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
    n = trace.length
    # Enough clocks for the last element to cross the array, and for OS the last
    # result to reach the north edge (a fixed count, so a bug cannot hang the run)
    stream_clocks = n + rows + cols + 1
    drain_clocks = 2 * rows + 1 if dataflow == OS else 0

    def snapshot(ctx, cycle, phase):
        def grid(name):
            return np.array(
                [[ctx.get(getattr(pes[i][j], name)) for j in range(cols)] for i in range(rows)]
            )

        trace.frames.append(
            Frame(
                cycle,
                phase,
                grid("out_valid").astype(bool),
                grid("in_shift").astype(bool),
                grid("out_a"),
                grid("out_b"),
                grid("out_psum"),
                grid("out_w"),
                grid("out_c"),
                grid("out_c_valid").astype(bool),
            )  # fmt: skip
        )

    def drive(ctx, **values):
        for name, value in values.items():
            ctx.set(getattr(dut, name), int(value))

    def edge_inputs(t):
        """Every edge input in stream cycle ``t``."""
        values = {}
        if dataflow == OS:
            for i in range(rows):
                k = t - stream_start(i)
                on = 0 <= k < n
                values.update({
                    f"in_a_{i}": A[i, k] if on else 0, f"in_valid_{i}": on,
                    f"in_first_{i}": on and k == 0, f"in_last_{i}": on and k == n - 1,
                })  # fmt: skip
            for j in range(cols):
                k = t - stream_start(j)
                values[f"in_b_{j}"] = B[k, j] if 0 <= k < n else 0
        else:
            for r in range(rows):
                m = t - stream_start(r)
                on = 0 <= m < n
                values.update({f"in_a_{r}": A[m, r] if on else 0, f"in_valid_{r}": on})
        return values

    async def run(ctx):
        drive(ctx, in_control_dataflow=dataflow)
        if dataflow == WS:
            drive(ctx, in_shift=1)
            for step, r in enumerate(reversed(range(rows))):
                drive(ctx, **{f"in_w_{j}": B[r, j] for j in range(cols)})
                snapshot(ctx, step - rows, "preload")
                await ctx.tick()
            drive(ctx, in_shift=0)
        out = {j: [] for j in range(cols)}
        for t in range(stream_clocks + drain_clocks):
            drive(ctx, **edge_inputs(t))
            snapshot(ctx, t, "stream" if t < stream_clocks else "drain")
            for j in range(cols):
                if dataflow == OS and ctx.get(getattr(dut, f"out_c_valid_{j}")):
                    i = len(out[j])
                    out[j].append(ctx.get(getattr(dut, f"out_c_{j}")))
                    trace.result_clocks[(i, j)] = t
                if dataflow == WS and ctx.get(getattr(dut, f"out_psum_valid_{j}")):
                    out[j].append(ctx.get(getattr(dut, f"out_psum_{j}")))
            await ctx.tick()
        depth = min(len(v) for v in out.values())
        trace.C = np.array([[out[j][r] for j in range(cols)] for r in range(depth)]).reshape(
            depth, cols
        )

    sim.add_clock(1e-6)
    sim.add_testbench(run)
    sim.run()
    return trace


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell_text(trace: Trace, frame: Frame, i: int, j: int) -> tuple[str, str]:
    """(label, kind) for one PE in one frame; kind is idle/active/mismatch/shift."""
    if frame.phase == "preload":
        return f"w{frame.w[i, j]}", "shift"
    if frame.c_valid[i, j]:  # an OS result on its way north
        return f"c{frame.c[i, j]}", "shift"
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
        "cN: a result moving north; thick lines: PEArray boundaries (every PE link is registered)",
    ]
    n_frames = len(trace.frames)
    total = n_frames * seconds_per_frame
    n_index = trace.length
    mode = "output-stationary" if trace.dataflow == OS else "stationary (WS / IS)"
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
            f"{'  (shift)' if f.phase == 'preload' else ''}</text>"
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
    # PEArray boundaries (structural: every PE link is registered)
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
