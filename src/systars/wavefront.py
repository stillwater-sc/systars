"""
Wavefront tracing of the systolic array RTL, and the animations drawn from it.

``record`` runs the real ``SystolicArray`` in Amaranth simulation, driven the way
the L1 stream buffers drive it (lane ``i``'s stream starts one clock after lane
``i - 1``'s), records what every PE holds in every clock, and ``Trace.check``
checks it against the wavefront the design promises. Every PE registers its
inputs, so the wavefront advances one PE per clock in every direction:

- Output-stationary: A[i, k] and B[k, j] meet at PE(i, j) in clock
  ``s + 1 + i + j + k`` (``s``: the clock the tile's stream starts), with
  ``valid``. After the last product the results move north, one PE per clock:
  PE(i, j)'s result leaves the north edge of column ``j`` in clock
  ``s + K + j + 2 i`` (``SystolicArray.os_result_clock``).
- Stationary (WS; IS is the same fabric mode): element ``m`` of the stream meets
  W[r, j] at PE(r, j) in clock ``s + 1 + r + j + m``, and leaves with the partial
  sum ``A[m, :r+1] @ W[:r+1, j]``; it leaves the south edge of column ``j`` in clock
  ``s + R + j + m`` (``SystolicArray.stationary_result_clock``).

A run is one tile or several back to back, started on the Streamer's schedule
(``streamer_starts``, which ``tests/unit/test_execute_controller.py`` checks
against the Streamer itself): with ``K >= 2R - 1`` (OS) or ``M >= R`` (stationary)
the tiles stream with no bubble between them.

Every cell is checked by value (the RTL's ``a``, ``b``, ``w`` and partial sums
against the operands), so any operands work, at any array size: the element a PE
holds is known from the schedule and confirmed by the values.

``render_svg`` draws a trace in one of three views, for the progression on the
wavefront page (``docs/wavefront-animation.md``):

- ``value``: the operands and running sums in every PE, and C filling in beside
  the array (stage 1, 2x2: the reduction);
- ``index``: which ``k`` (OS) or ``m`` (stationary) meets at each PE, and the L1
  stream buffers feeding the edges (stage 2, 4x4: the stagger);
- ``front``: cells coloured by ``k`` / ``m`` with no text, results moving north as
  dots, and the fraction of busy PEs over time (stage 3, 16x16: the schedule).

The SVG animates each cell's colour and text over the frames (SMIL, one
``<animate>`` per element), so its size grows with the cells and their changes, not
with cells x frames. It carries its own light and dark colours
(``prefers-color-scheme``) and a colour-blind-safe palette (viridis).
``STAGES`` and ``stage_svg`` make the animations the docs embed
(``scripts/gen_wavefronts.py``, ``just wavefronts``).
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

#: A west/north edge entry: (tile, element index), or (WEIGHTS, row) for a weight
#: shifting in during a stationary preload
WEIGHTS = -1


def stream_start(lane: int) -> int:
    """The clock lane ``lane``'s stream starts, relative to its tile (the L1 stagger)."""
    return lane


def streamer_starts(
    dataflow: int, rows: int, cols: int, lengths: list[int], first: int = 0
) -> list[int]:
    """The clocks the Streamer starts each tile's stream (lane 0), back to back.

    The next tile starts ``max(K, R, C)`` clocks after the last (OS; ``max(M, R)``
    stationary), and an OS reduction only once its results will leave ``2R - 1``
    clocks after the previous reduction's, so results never meet on a north link
    (``controller/execute.py``, "Schedule"). Every tile here is a whole reduction,
    streamed as one command or as commands of at least ``max(R, C)`` elements each
    (shorter chained commands each take ``max(R, C)`` clocks).
    """
    starts: list[int] = []
    for n, length in enumerate(lengths):
        if n == 0:
            starts.append(first)
            continue
        prev = lengths[n - 1]
        if dataflow == OS:
            gap = max(prev, rows, cols, prev - length + 2 * rows - 1)
        else:
            gap = max(prev, rows)
        starts.append(starts[-1] + gap)
    return starts


@dataclass
class Tile:
    """One stream through the fabric: OS ``A`` (R x K) and ``B`` (K x C); stationary
    ``A`` (M x R) against the trace's weights."""

    start: int
    length: int  # K (OS) or M (stationary)
    A: np.ndarray
    B: np.ndarray | None = None


@dataclass
class Frame:
    """What every PE holds in one clock (arrays are rows x cols), and the edges."""

    cycle: int
    phase: str  # "preload" (stationary weights), "stream", or "drain" (OS results north)
    valid: np.ndarray
    shift: np.ndarray
    a: np.ndarray
    b: np.ndarray
    psum_out: np.ndarray
    w: np.ndarray  # the stationary register
    c: np.ndarray  # the OS result each PE sends north this clock
    c_valid: np.ndarray
    west: list = field(default_factory=list)  # per row: (tile, k) entering, or None
    north: list = field(default_factory=list)  # per column: (tile, k) / (WEIGHTS, r), or None
    west_values: list = field(default_factory=list)
    north_values: list = field(default_factory=list)


@dataclass
class Trace:
    """A recorded run of one or more tiles (stationary: against the weights ``W``)."""

    config: SystolicConfig
    dataflow: int
    tiles: list[Tile]
    W: np.ndarray | None
    C: np.ndarray  # the results as the RTL produced them, tile after tile
    frames: list[Frame] = field(default_factory=list)
    #: (tile, row, col) -> the clock that result left the array (OS: north edge,
    #: row = PE row; stationary: south edge, row = stream element m)
    result_clocks: dict = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return self.config.grid_rows * self.config.tile_rows

    @property
    def cols(self) -> int:
        return self.config.grid_cols * self.config.tile_cols

    @property
    def length(self) -> int:
        """K (OS) or M (stationary) of the first tile (``0`` for no tiles)."""
        return self.tiles[0].length if self.tiles else 0

    def meeting(self, cycle: int, i: int, j: int) -> tuple[int, int] | None:
        """The (tile, k or m) the schedule puts at PE(i, j) in stream clock ``cycle``."""
        for n, tile in enumerate(self.tiles):
            index = cycle - 1 - i - j - tile.start
            if 0 <= index < tile.length:
                return n, index
        return None

    def expected(self) -> np.ndarray:
        """``A @ B`` for every tile, stacked as the RTL produces them."""
        if self.dataflow == OS:
            parts = [t.A @ t.B for t in self.tiles if t.length]
        else:
            parts = [t.A @ self.W for t in self.tiles if t.length]
        return np.vstack(parts) if parts else np.zeros((0, self.cols), dtype=np.int64)

    def expected_result_clocks(self) -> dict:
        clocks = {}
        for n, t in enumerate(self.tiles):
            if self.dataflow == OS:
                for i in range(self.rows):
                    for j in range(self.cols):
                        clocks[(n, i, j)] = SystolicArray.os_result_clock(t.start, t.length, i, j)
            else:
                for m in range(t.length):
                    for j in range(self.cols):
                        clocks[(n, m, j)] = t.start + self.rows + j + m
        return clocks

    def cell_problem(self, frame: Frame, i: int, j: int) -> str | None:
        """How PE(i, j) departs from the prediction in a stream ``frame`` (None: it doesn't)."""
        where = f"cycle {frame.cycle}, PE({i},{j})"
        meet = self.meeting(frame.cycle, i, j)
        if bool(frame.valid[i, j]) != (meet is not None):
            return f"{where}: valid={int(frame.valid[i, j])}, expected {int(meet is not None)}"
        if meet is None:
            return None
        n, idx = meet
        A = self.tiles[n].A
        if self.dataflow == OS:
            B = self.tiles[n].B
            assert B is not None
            if (frame.a[i, j], frame.b[i, j]) != (A[i, idx], B[idx, j]):
                return (
                    f"{where}: a={frame.a[i, j]} b={frame.b[i, j]}, expected "
                    f"A[{i},{idx}]={A[i, idx]} B[{idx},{j}]={B[idx, j]}"
                )
            return None
        W = self.W
        assert W is not None
        psum = int(A[idx, : i + 1] @ W[: i + 1, j])
        if frame.a[i, j] != A[idx, i] or frame.psum_out[i, j] != psum or frame.w[i, j] != W[i, j]:
            return (
                f"{where}: a={frame.a[i, j]} psum={frame.psum_out[i, j]} w={frame.w[i, j]}, "
                f"expected A[{idx},{i}]={A[idx, i]} psum={psum} W[{i},{j}]={W[i, j]}"
            )
        return None

    def check(self) -> list[str]:
        """Every place the RTL departs from the predicted wavefront (empty: correct).

        Checks every PE in every stream clock, the clock every result leaves the
        array, and the results. Preload and drain frames are recorded for drawing.
        """
        problems = []
        for f in self.frames:
            if f.phase != "stream":
                continue
            for i in range(self.rows):
                for j in range(self.cols):
                    problem = self.cell_problem(f, i, j)
                    if problem:
                        problems.append(problem)
        if any(t.length for t in self.tiles):
            expected = self.expected_result_clocks()
            if self.result_clocks != expected:
                missing = len(expected) - len(self.result_clocks)
                problems.append(f"results left the array at the wrong clocks ({missing} missing)")
        expected_C = self.expected()
        if self.C.shape != expected_C.shape or not np.array_equal(self.C, expected_C):
            problems.append("result differs from A @ B")
        return problems

    def observed(self, frame: Frame, i: int, j: int) -> tuple[int | None, bool]:
        """The k (OS) or m (stationary) at PE(i, j), and whether the RTL agrees.

        (None, True) for a PE that is idle, as predicted.
        """
        if frame.phase != "stream":
            return None, True
        meet = self.meeting(frame.cycle, i, j)
        agree = self.cell_problem(frame, i, j) is None
        if meet is None:
            return None, agree
        return meet[1], agree

    def utilization(self, frame: Frame) -> float:
        """The fraction of PEs doing a multiply-accumulate in ``frame``."""
        return float(frame.valid.mean()) if frame.phase != "preload" else 0.0

    def phase_label(self, frame: Frame) -> str:
        """preload, fill (the front crossing the array), steady, or drain."""
        if frame.phase == "preload":
            return "preload"
        entries = [f.cycle for f in self.frames if f.phase != "preload" and _feeding(f)]
        if not entries or frame.cycle > max(entries):
            return "drain"
        if frame.cycle < min(t.start for t in self.tiles) + self.rows + self.cols - 1:
            return "fill"
        return "steady"


def _feeding(frame: Frame) -> bool:
    """Whether an L1 stream buffer feeds the array an element in ``frame``."""
    return any(e is not None for e in frame.west) or any(
        e is not None and e[0] != WEIGHTS for e in frame.north
    )


def operands(
    config: SystolicConfig,
    dataflow: int,
    length: int,
    *,
    tiles: int = 1,
    small: bool = False,
    seed: int = 0,
):
    """Random operands for ``record``: OS ``(As, Bs)``, stationary ``(As, W)``.

    ``small``: single digits 1-9 (readable sums, for the value view); otherwise
    int8. One tile returns arrays, more return lists of arrays.
    """
    rows = config.grid_rows * config.tile_rows
    cols = config.grid_cols * config.tile_cols
    rng = np.random.default_rng(seed)
    lo, hi = (1, 10) if small else (-128, 128)
    if dataflow == OS:
        As = [rng.integers(lo, hi, (rows, length)) for _ in range(tiles)]
        Bs = [rng.integers(lo, hi, (length, cols)) for _ in range(tiles)]
        return (As[0], Bs[0]) if tiles == 1 else (As, Bs)
    W = rng.integers(lo, hi, (rows, cols))
    As = [rng.integers(lo, hi, (length, rows)) for _ in range(tiles)]
    return (As[0] if tiles == 1 else As), W


def record(config: SystolicConfig, dataflow: int, A, B, *, starts=None) -> Trace:
    """Run ``A @ B`` on the RTL array and record every PE in every clock.

    OS: ``A`` is R x K and ``B`` K x C, or lists of them (one reduction per tile).
    Stationary: ``A`` is M x R (or a list of them) and ``B`` the R x C weights.
    ``starts`` (default: ``streamer_starts``) are the clocks each tile's stream
    starts. Stream clocks are numbered from 0; preload clocks are negative.
    """
    as_list = isinstance(A, (list, tuple))
    As = [np.asarray(a) for a in (A if as_list else [A])]
    if dataflow == OS:
        Bs: list = [np.asarray(b) for b in (B if as_list else [B])]
        if len(Bs) != len(As):
            raise ValueError("OS needs one B tile per A tile")
        lengths = [a.shape[1] for a in As]
        W = None
    else:
        Bs = [None] * len(As)
        lengths = [a.shape[0] for a in As]
        W = np.asarray(B)
    dut = SystolicArray(config)
    rows, cols = dut.rows, dut.cols
    if starts is None:
        starts = streamer_starts(dataflow, rows, cols, lengths)
    tiles = [Tile(s, n, a, b) for s, n, a, b in zip(starts, lengths, As, Bs, strict=True)]
    trace = Trace(config, dataflow, tiles, W, np.zeros((0, cols), dtype=np.int64))
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
    end = max((t.start + t.length for t in tiles), default=0)
    # Enough clocks for the last element to cross the array, and for OS the last
    # result to reach the north edge (a fixed count, so a bug cannot hang the run)
    stream_clocks = end + rows + cols + 1
    drain_clocks = 2 * rows + 1 if dataflow == OS else 0

    def snapshot(ctx, cycle, phase, edges):
        def grid(name):
            return np.array(
                [[ctx.get(getattr(pes[i][j], name)) for j in range(cols)] for i in range(rows)]
            )

        west, north, west_values, north_values = edges
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
                west,
                north,
                west_values,
                north_values,
            )  # fmt: skip
        )

    def drive(ctx, **values):
        for name, value in values.items():
            ctx.set(getattr(dut, name), int(value))

    def entry(t, lane):
        """The (tile, k) lane ``lane``'s L1 buffer feeds in clock ``t``, if any."""
        for n, tile in enumerate(tiles):
            k = t - tile.start - stream_start(lane)
            if 0 <= k < tile.length:
                return n, k
        return None

    def edge_inputs(t):
        """Every edge input in stream clock ``t``, and what each lane carries."""
        values, west, north, west_values, north_values = {}, [], [], [], []
        for i in range(rows):
            e = entry(t, i)
            west.append(e)
            n, k = e if e else (0, 0)
            if dataflow == OS:
                a = tiles[n].A[i, k] if e else 0
                values.update({
                    f"in_a_{i}": a, f"in_valid_{i}": e is not None,
                    f"in_first_{i}": e is not None and k == 0,
                    f"in_last_{i}": e is not None and k == tiles[n].length - 1,
                })  # fmt: skip
            else:
                a = tiles[n].A[k, i] if e else 0
                values.update({f"in_a_{i}": a, f"in_valid_{i}": e is not None})
            west_values.append(int(a))
        for j in range(cols):
            e = entry(t, j) if dataflow == OS else None
            north.append(e)
            b = tiles[e[0]].B[e[1], j] if e else 0
            if dataflow == OS:
                values[f"in_b_{j}"] = b
            north_values.append(int(b))
        return values, (west, north, west_values, north_values)

    async def run(ctx):
        drive(ctx, in_control_dataflow=dataflow)
        if dataflow == WS:
            drive(ctx, in_shift=1)
            for step, r in enumerate(reversed(range(rows))):
                drive(ctx, **{f"in_w_{j}": W[r, j] for j in range(cols)})
                edges = ([None] * rows, [(WEIGHTS, r)] * cols, [0] * rows, [int(v) for v in W[r]])
                snapshot(ctx, step - rows, "preload", edges)
                await ctx.tick()
            drive(ctx, in_shift=0)
        # The order results leave each column: (tile, row) per result
        if dataflow == OS:
            order = [(n, i) for n, t in enumerate(tiles) if t.length for i in range(rows)]
        else:
            order = [(n, m) for n, t in enumerate(tiles) for m in range(t.length)]
        out = {j: [] for j in range(cols)}
        for t in range(stream_clocks + drain_clocks):
            values, edges = edge_inputs(t)
            drive(ctx, **values)
            snapshot(ctx, t, "stream" if t < stream_clocks else "drain", edges)
            for j in range(cols):
                kind = "c" if dataflow == OS else "psum"
                if ctx.get(getattr(dut, f"out_{kind}_valid_{j}")):
                    q = len(out[j])
                    out[j].append(ctx.get(getattr(dut, f"out_{kind}_{j}")))
                    if q < len(order):
                        tile, row = order[q]
                        trace.result_clocks[(tile, row, j)] = t
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
# Text rendering
# ---------------------------------------------------------------------------


def _index_label(trace: Trace, frame: Frame, i: int, j: int) -> tuple[str, str]:
    """(label, kind) for one PE in one frame; kind is idle/active/mismatch/shift."""
    if frame.phase == "preload":
        return f"w{frame.w[i, j]}", "shift"
    index, agree = trace.observed(frame, i, j)
    if not agree:
        return "X", "mismatch"
    if index is not None:
        return ("k" if trace.dataflow == OS else "m") + str(index), "active"
    if frame.c_valid[i, j]:  # an OS result on its way north
        return f"c{frame.c[i, j]}", "shift"
    return ".", "idle"


def render_text(trace: Trace, width: int = 7) -> str:
    """The frames as text, one grid per clock; '|' and '-' mark PEArray boundaries."""
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
                cells.append(_index_label(trace, f, i, j)[0].center(width))
            out.append("    " + " ".join(cells))
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------

# viridis (colour-blind safe), from 0.25 to 1.0, lightened for black cell text
_VIRIDIS = [
    "#440154", "#482475", "#414487", "#355f8d", "#2a788e", "#21918c",
    "#22a884", "#44bf70", "#7ad151", "#bddf26", "#fde725",
]  # fmt: skip
_SHIFT = "#b8c4d6"  # a weight shifting in, a result moving north
_MISMATCH = "#d62728"

_STYLE = """
.bg{fill:#ffffff}.fg{fill:#1b1f24}.muted{fill:#59636e}.idle{fill:#eceff3}
.line{stroke:#1b1f24}.cellline{stroke:#ffffff}.plot{stroke:#2a788e}
.cell{fill:#111;stroke:#fff;stroke-width:3px;paint-order:stroke;stroke-linejoin:round}
.dot{fill:#1b1f24;stroke:#ffffff;stroke-width:1.5px}
@media (prefers-color-scheme: dark){
.bg{fill:#17181c}.fg{fill:#e6e8eb}.muted{fill:#9aa4ae}.idle{fill:#2b3038}
.line{stroke:#e6e8eb}.cellline{stroke:#17181c}.plot{stroke:#7ad151}}
"""


def _mix(c1: str, c2: str, t: float) -> list[float]:
    a = [int(c1[k : k + 2], 16) for k in (1, 3, 5)]
    b = [int(c2[k : k + 2], 16) for k in (1, 3, 5)]
    return [x + (y - x) * t for x, y in zip(a, b, strict=True)]


def index_colour(index: int, n: int) -> str:
    """The fill for element ``index`` of ``n``: one colour per wavefront (viridis)."""
    t = 0.25 + 0.75 * (index / (n - 1) if n > 1 else 0.5)
    pos = t * (len(_VIRIDIS) - 1)
    k = min(int(pos), len(_VIRIDIS) - 2)
    rgb = _mix(_VIRIDIS[k], _VIRIDIS[k + 1], pos - k)
    rgb = [x + (255 - x) * 0.3 for x in rgb]  # lighten, so black text reads on it
    return "#" + "".join(f"{round(x):02x}" for x in rgb)


class _Timeline:
    """Frame start times, and SMIL ``<animate>`` elements over them."""

    def __init__(self, durations: list[float]):
        self.total = sum(durations)
        self.starts = np.concatenate([[0.0], np.cumsum(durations)[:-1]]).tolist()
        # Enough decimals for keyTimes to keep the shortest frame apart from the next
        shortest = min(durations) / self.total
        self.digits = max(2, int(np.ceil(-np.log10(shortest))) + 1)

    def animate(self, attr: str, per_frame: list[str]) -> str:
        """An ``<animate>`` stepping ``attr`` through ``per_frame`` ('' if constant)."""
        changes = [(f, v) for f, v in enumerate(per_frame) if f == 0 or v != per_frame[f - 1]]
        if len(changes) == 1:
            return ""
        times = ";".join(
            "0" if f == 0 else f"{self.starts[f] / self.total:.{self.digits}f}" for f, _ in changes
        )
        values = ";".join(v for _, v in changes)
        return (
            f'<animate attributeName="{attr}" calcMode="discrete" dur="{self.total:.2f}s" '
            f'repeatCount="indefinite" keyTimes="{times}" values="{values}"/>'
        )

    def shown(self, per_frame: list[bool]) -> tuple[str, str]:
        """(initial opacity, animate) for an element shown in some frames."""
        values = ["1" if s else "0" for s in per_frame]
        return values[0], self.animate("opacity", values)


def _texts(timeline: _Timeline, x, y, per_frame: list[str], cls: str, size: int, **attrs) -> str:
    """One ``<text>`` per distinct string, each shown in the frames that have it."""
    extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    parts = []
    for s in dict.fromkeys(per_frame):
        if not s:
            continue
        vis, anim = timeline.shown([p == s for p in per_frame])
        parts.append(
            f'<text x="{x:g}" y="{y:g}" font-size="{size}" class="{cls}" '
            f'opacity="{vis}"{extra}>{html.escape(s)}{anim}</text>'
        )
    return "".join(parts)


VIEWS = ("value", "index", "front")


def default_view(trace: Trace) -> str:
    """value up to 2x2, index up to 8x8, front beyond."""
    size = max(trace.rows, trace.cols)
    return "value" if size <= 2 else "index" if size <= 8 else "front"


def render_svg(
    trace: Trace,
    *,
    view: str = "auto",
    title: str | None = None,
    seconds_per_frame: float | None = None,
    utilization: bool | None = None,
) -> str:
    """An animated SVG (SMIL) of ``trace``, looping, in a ``value``, ``index`` or
    ``front`` view (``auto``: by size, ``default_view``)."""
    if view == "auto":
        view = default_view(trace)
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS} or 'auto', not {view!r}")
    return _Renderer(trace, view, title, seconds_per_frame, utilization).svg()


class _Renderer:
    def __init__(self, trace, view, title, seconds_per_frame, utilization):
        self.t, self.view = trace, view
        self.os = trace.dataflow == OS
        self.rows, self.cols = trace.rows, trace.cols
        self.frames = trace.frames
        self.idx = "k" if self.os else "m"
        self.cell = {"value": 104, "index": 54, "front": max(14, min(30, 480 // self.cols))}[view]
        self.utilization = view == "front" if utilization is None else utilization
        self.problems = trace.check()
        self.notes = self._notes()
        base = seconds_per_frame or {"value": 1.4, "index": 0.8, "front": 0.25}[view]
        durations = [base * (4 if n in self.notes else 1) for n in range(len(self.frames))]
        if durations:
            durations[-1] = base * 6  # hold the end of the loop
        self.tl = _Timeline(durations or [1.0])
        self.title = title or self._title()

    # -- what to say -------------------------------------------------------

    def _title(self) -> str:
        mode = "Output-stationary" if self.os else "Stationary (WS / IS)"
        cfg = self.t.config
        shape = f"{self.rows}x{self.cols} PEs"
        if cfg.tile_rows > 1 or cfg.tile_cols > 1:
            shape += (
                f" ({cfg.grid_rows}x{cfg.grid_cols} PEArrays of {cfg.tile_rows}x{cfg.tile_cols})"
            )
        n = len(self.t.tiles)
        lengths = sorted({t.length for t in self.t.tiles})
        length = f"{'K' if self.os else 'M'} = {', '.join(map(str, lengths))}"
        tiles = f", {n} tiles back to back" if n > 1 else ""
        return f"{mode}, {shape}, {length}{tiles}"

    def _notes(self) -> dict[int, str]:
        """Frames to dwell on, and what to say about them."""
        t = self.t
        notes: dict[int, str] = {}
        by_cycle = {f.cycle: n for n, f in enumerate(self.frames) if f.phase != "preload"}
        tiles = [x for x in t.tiles if x.length]
        if not tiles:
            return notes
        first = tiles[0].start + 1
        if first in by_cycle:
            what = "A[0,0] meets B[0,0]" if self.os else "A[0,0] meets W[0,0]"
            notes[by_cycle[first]] = f"{what} at PE(0,0): the first product"
        for n in range(1, len(t.tiles)):
            prev, cur = t.tiles[n - 1], t.tiles[n]
            c = cur.start + 1
            if c in by_cycle:
                gap = cur.start - prev.start - prev.length
                follow = (
                    "with no bubble"
                    if gap <= 0
                    else f"{gap} clocks later (the north drain: 2R - 1 between reductions)"
                )
                notes[by_cycle[c]] = f"tile {n + 1} follows tile {n} {follow}"
        if t.result_clocks:
            clock = min(t.result_clocks.values())
            if clock in by_cycle and by_cycle[clock] not in notes:
                notes[by_cycle[clock]] = (
                    "PE(0,0)'s result leaves the north edge for the L1 result buffer"
                    if self.os
                    else "the first partial sums leave the south edge for the L1 result buffer"
                )
        return notes

    def caption(self, f: Frame) -> str:
        phase = self.t.phase_label(f)
        if phase == "preload":
            return f"clock {f.cycle}  preload: the weights shift in, one row per clock"
        busy = f"  {self.t.utilization(f):.0%} of PEs busy" if self.view == "front" else ""
        return f"clock {f.cycle}  {phase}{busy}"

    def legend(self) -> list[str]:
        v = self.view
        if v == "value":
            if self.os:
                return [
                    "a·b: the operands in the PE's input registers; c: its running sum",
                    "↑: a result moving north to the L1 result buffer, one PE per clock",
                ]
            return [
                "w: the stationary weight; a: the streamed element; Σ: the partial sum leaving south",
            ]
        lines = [
            f"{self.idx}N: element N meets there (one colour per {self.idx}: a wavefront); "
            "edges: the L1 stream buffers",
        ]
        if v == "front":
            lines = [
                f"colour: the {self.idx} meeting at each PE (one colour per wavefront); "
                + ("dots: results moving north" if self.os else "edges: the L1 stream buffers"),
            ]
        if self.os and v == "index":
            lines.append("cN: a result moving north; thick lines: PEArray boundaries")
        elif self.t.config.tile_rows > 1 or self.t.config.tile_cols > 1:
            lines.append(
                "thick lines: PEArray boundaries (structural: every PE link is registered)"
            )
        return lines

    # -- per-cell content ----------------------------------------------------

    def fill(self, f: Frame, i: int, j: int) -> str:
        if f.phase == "preload":
            return _SHIFT if f.shift[i, j] else "none"
        index, agree = self.t.observed(f, i, j)
        if not agree:
            return _MISMATCH
        if index is not None:
            n, _ = self.t.meeting(f.cycle, i, j)
            return index_colour(index, self.t.tiles[n].length)
        if f.c_valid[i, j] and self.view != "front":
            return _SHIFT
        return "none"

    def lines(self, f: Frame, i: int, j: int, sums) -> list[str]:
        """Up to three text lines for one cell."""
        t = self.t
        if self.view == "index":
            label, kind = _index_label(t, f, i, j)
            return ["" if kind == "idle" else label]
        # value view
        if f.phase == "preload":
            return ["", f"w {f.w[i, j]}", "↓"] if f.shift[i, j] else ["", "", ""]
        index, agree = t.observed(f, i, j)
        if not agree:
            return ["", "X", ""]
        if self.os:
            if index is not None:
                return [f"{f.a[i, j]}·{f.b[i, j]}", f"c = {sums[i][j]}", f"{self.idx} = {index}"]
            if f.c_valid[i, j]:
                return ["", f"↑ {f.c[i, j]}", ""]
            return ["", "", ""]
        w = f"w {f.w[i, j]}"
        if index is not None:
            return [w, f"a {f.a[i, j]}", f"Σ {f.psum_out[i, j]}"]
        return [w, "", ""]

    # -- drawing -------------------------------------------------------------

    def svg(self) -> str:
        t, cell, frames, tl = self.t, self.cell, self.frames, self.tl
        rows, cols = self.rows, self.cols
        margin, char_w = 20, 7.2
        strip = cell if self.view != "front" else max(8, cell // 2)
        gap = 6
        header = 74
        x0 = margin + strip + gap
        y0 = header + strip + gap
        array_w, array_h = cols * cell, rows * cell
        panel_x = x0 + array_w + 2 * margin
        panel = self._panel_lines() if self.view == "value" else []
        panel_w = max((len(s) for s, *_ in panel), default=0) * 8.4
        panel_w = max(panel_w, 26 * 8) if panel else 0  # the verdict under C
        util_h = 56 if self.utilization else 0
        legend = self.legend()
        text_w = max(len(s) for s in [self.title, *legend, *self.notes.values(), ""]) * char_w
        width = int(
            max(panel_x + panel_w if panel else x0 + array_w + margin, margin + text_w + margin)
        )
        legend_y = y0 + array_h + util_h + 26
        height = int(max(legend_y + 16 * len(legend), y0 + len(panel) * 18 + 10) + margin / 2)

        out = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" xml:space="preserve" '
            'font-family="ui-monospace, Menlo, Consolas, monospace">',
            f"<style>{_STYLE}</style>",
            f'<rect class="bg" width="{width}" height="{height}" rx="6"/>',
            f'<text x="{margin}" y="22" font-size="14" class="fg" font-weight="600">'
            f"{html.escape(self.title)}</text>",
            _texts(tl, margin, 42, [self.caption(f) for f in frames], "fg", 13),
            _texts(tl, margin, 60, [self.notes.get(n, "") for n in range(len(frames))], "muted", 12, font_style="italic"),
        ]  # fmt: skip

        # Running sums (OS value view), from the RTL's own operands
        sums_per_frame = []
        acc = [[0] * cols for _ in range(rows)]
        for f in frames:
            for i in range(rows):
                for j in range(cols):
                    if f.phase == "stream" and f.valid[i, j]:
                        meet = t.meeting(f.cycle, i, j)
                        product = int(f.a[i, j]) * int(f.b[i, j])
                        acc[i][j] = product if meet and meet[1] == 0 else acc[i][j] + product
            sums_per_frame.append([row[:] for row in acc])

        # The array: a static idle cell under an animated fill, and the text lines
        font = {"value": 15, "index": 12, "front": 0}[self.view]
        out.append(
            f'<rect class="idle" x="{x0}" y="{y0}" width="{cols * cell}" height="{rows * cell}"/>'
        )
        # Cells whose fills step through the same colours share one animation (on an
        # anti-diagonal they do): exact, and it keeps a large array's SVG small
        groups: dict[tuple, list] = {}
        for i in range(rows):
            for j in range(cols):
                fills = tuple(self.fill(f, i, j) for f in frames)
                groups.setdefault(fills, []).append((x0 + j * cell, y0 + i * cell))
        for fills, cells in groups.items():
            rects = "".join(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}"/>' for x, y in cells
            )
            out.append(
                f'<g fill="{fills[0]}" class="cellline" stroke-width="1">'
                f"{tl.animate('fill', list(fills))}{rects}</g>"
            )
        for i in range(rows):
            for j in range(cols):
                x, y = x0 + j * cell, y0 + i * cell
                if self.view == "front" and self.os:
                    vis, anim = tl.shown([bool(f.c_valid[i, j]) for f in frames])
                    if anim or vis == "1":
                        out.append(
                            f'<circle class="dot" cx="{x + cell / 2}" cy="{y + cell / 2}" '
                            f'r="{cell / 5:.1f}" opacity="{vis}">{anim}</circle>'
                        )
                if font:
                    per_line = list(
                        zip(
                            *[self.lines(f, i, j, sums_per_frame[n]) for n, f in enumerate(frames)],
                            strict=True,
                        )
                    )
                    offsets = (
                        [0] if len(per_line) == 1 else [-cell * 0.28, 0.04 * cell, cell * 0.33]
                    )
                    for line, dy in zip(per_line, offsets, strict=False):
                        size = font if len(per_line) == 1 or dy == offsets[1] else font - 3
                        out.append(_texts(tl, x + cell / 2, y + cell / 2 + dy + size * 0.35,
                                          list(line), "cell", size, text_anchor="middle"))  # fmt: skip

        out += self._edges(x0, y0, strip, gap, font)
        out += self._boundaries(x0, y0)
        if self.utilization:
            out += self._utilization(x0, y0 + array_h + 14, array_w, util_h - 18)
        if panel:
            out += self._panel(panel, panel_x, y0)
        for n, text in enumerate(legend):
            out.append(
                f'<text x="{margin}" y="{legend_y + 16 * n}" font-size="12" class="muted">'
                f"{html.escape(text)}</text>"
            )
        out.append("</svg>")
        return "\n".join(s for s in out if s) + "\n"

    def _edges(self, x0, y0, strip, gap, font) -> list[str]:
        """The L1 stream buffers' lanes: what each feeds the array in each clock."""
        t, tl, frames, cell = self.t, self.tl, self.frames, self.cell
        out = []

        def lane(x, y, w, h, entries, values, prefix):
            fills, labels = [], []
            for e, v in zip(entries, values, strict=True):
                if e is None:
                    fills.append("none")
                    labels.append("")
                elif e[0] == WEIGHTS:
                    fills.append(_SHIFT)
                    labels.append(f"w {v}" if self.view == "value" else "")
                else:
                    fills.append(index_colour(e[1], t.tiles[e[0]].length))
                    labels.append(f"{prefix} {v}" if self.view == "value" else f"{self.idx}{e[1]}")
            out.append(
                f'<rect class="idle" x="{x}" y="{y}" width="{w}" height="{h}" opacity="0.6"/>'
            )
            out.append(
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fills[0]}" '
                f'class="cellline">{tl.animate("fill", fills)}</rect>'
            )
            if font:
                size = font - 2
                out.append(_texts(tl, x + w / 2, y + h / 2 + size * 0.35, labels, "cell", size,
                                  text_anchor="middle"))  # fmt: skip

        for i in range(self.rows):
            lane(x0 - gap - strip, y0 + i * cell, strip, cell,
                 [f.west[i] if f.west else None for f in frames],
                 [f.west_values[i] if f.west_values else 0 for f in frames], "a")  # fmt: skip
        if self.os or t.W is not None:
            for j in range(self.cols):
                lane(x0 + j * cell, y0 - gap - strip, cell, strip,
                     [f.north[j] if f.north else None for f in frames],
                     [f.north_values[j] if f.north_values else 0 for f in frames], "b")  # fmt: skip
        return out

    def _boundaries(self, x0, y0) -> list[str]:
        cfg, cell = self.t.config, self.cell
        out = []
        w, h = self.cols * cell, self.rows * cell
        out.append(
            f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" class="line" stroke-width="2"/>'
        )
        if cfg.tile_rows > 1:
            for tr in range(1, cfg.grid_rows):
                y = y0 + tr * cfg.tile_rows * cell
                out.append(
                    f'<line x1="{x0}" y1="{y}" x2="{x0 + w}" y2="{y}" class="line" stroke-width="2"/>'
                )
        if cfg.tile_cols > 1:
            for tc in range(1, cfg.grid_cols):
                x = x0 + tc * cfg.tile_cols * cell
                out.append(
                    f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y0 + h}" class="line" stroke-width="2"/>'
                )
        return out  # fmt: skip

    def _utilization(self, x, y, w, h) -> list[str]:
        """The fraction of busy PEs per clock, with a cursor on the current clock."""
        n = len(self.frames)
        if not n:
            return []
        step = w / n
        points = " ".join(
            f"{x + (k + 0.5) * step:.1f},{y + h - h * self.t.utilization(f):.1f}"
            for k, f in enumerate(self.frames)
        )
        cursor = [f"{x + k * step:.1f}" for k in range(n)]
        return [
            f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" class="line" stroke-width="1"/>',
            f'<polyline points="{points}" fill="none" class="plot" stroke-width="2"/>',
            f'<rect x="{cursor[0]}" y="{y}" width="{max(step, 1.5):.1f}" height="{h}" class="fg" '
            f'opacity="0.35">{self.tl.animate("x", cursor)}</rect>',
            f'<text x="{x}" y="{y - 3}" font-size="11" class="muted">PEs busy (0-100%) per clock</text>',
        ]  # fmt: skip

    def _panel_lines(self) -> list[tuple[str, ...]]:
        """The value view's side panel: the operands, then C (entries as they arrive)."""
        t = self.t
        lines: list[tuple[str, ...]] = []

        def matrix(name, M):
            lines.append((name,))
            width = max(len(str(int(v))) for v in M.flat) if M.size else 1
            for r in M:
                lines.append(("[" + " ".join(str(int(v)).rjust(width) for v in r) + "]",))

        tiles = [x for x in t.tiles if x.length]
        if not tiles:
            return lines
        A = np.vstack([x.A for x in tiles]) if not self.os else np.hstack([x.A for x in tiles])
        matrix("A" if self.os else "A (streamed)", A)
        if self.os:
            matrix("B", np.vstack([x.B for x in tiles]))
        else:
            matrix("W (stationary)", t.W)
        lines.append(("C = A @ B" if self.os else "C = A @ W", "C"))
        return lines

    def _panel(self, lines, x, y) -> list[str]:
        t, tl, frames = self.t, self.tl, self.frames
        out = []
        by_cycle = {f.cycle: n for n, f in enumerate(frames)}
        for n, line in enumerate(lines):
            out.append(
                f'<text x="{x}" y="{y + 14 + n * 18}" font-size="14" class="fg">'
                f"{html.escape(line[0])}</text>"
            )
        # C, entry by entry, each from the clock it leaves the array
        C = t.C
        expected = t.expected()
        base = y + 14 + len(lines) * 18
        width = max((len(str(int(v))) for v in expected.flat), default=1)
        rows_c = expected.shape[0]
        arrived = {}
        if self.os:
            for (n, i, j), clock in t.result_clocks.items():
                arrived[(n * self.rows + i, j)] = by_cycle.get(clock, len(frames) - 1)
        else:
            offset = np.cumsum([0] + [x.length for x in t.tiles])
            for (n, m, j), clock in t.result_clocks.items():
                arrived[(offset[n] + m, j)] = by_cycle.get(clock, len(frames) - 1)
        for r in range(rows_c):
            out.append(f'<text x="{x}" y="{base + r * 18}" font-size="14" class="muted">[</text>')
            for j in range(self.cols):
                value = int(C[r, j]) if r < C.shape[0] else None
                start = arrived.get((r, j), len(frames))
                shown = [k >= start for k in range(len(frames))]
                vis, anim = tl.shown(shown)
                right = x + 8.4 * (1 + j * (width + 1) + width)
                if value is not None:
                    out.append(
                        f'<text x="{right:.1f}" y="{base + r * 18}" font-size="14" class="fg" '
                        f'text-anchor="end" opacity="{vis}">{value}{anim}</text>'
                    )
            out.append(
                f'<text x="{x + 8.4 * (1 + self.cols * (width + 1) - 1):.1f}" y="{base + r * 18}" '
                f'font-size="14" class="muted">]</text>'
            )
        done = max(arrived.values(), default=len(frames) - 1)
        verdict = (
            ("C == A @ B, from the RTL" if self.os else "C == A @ W, from the RTL")
            if not self.problems
            else "the RTL departs from the prediction"
        )
        vis, anim = tl.shown([k >= done for k in range(len(frames))])
        out.append(
            f'<text x="{x}" y="{base + rows_c * 18 + 8}" font-size="13" class="fg" '
            f'font-weight="600" opacity="{vis}">{html.escape(verdict)}{anim}</text>'
        )
        return out


# ---------------------------------------------------------------------------
# The docs' progression
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    """One animation of the progression on the wavefront page."""

    number: int
    slug: str
    grid: tuple[int, int]
    tile: tuple[int, int]
    length: int  # K (OS) or M (stationary) per tile
    tiles: int
    view: str
    small: bool  # single-digit operands (readable sums)

    def config(self) -> SystolicConfig:
        return SystolicConfig(
            grid_rows=self.grid[0],
            grid_cols=self.grid[1],
            tile_rows=self.tile[0],
            tile_cols=self.tile[1],
        )

    def filename(self, dataflow: int) -> str:
        return f"{'os' if dataflow == OS else 'ws'}-{self.number}-{self.slug}.svg"


#: 2x2 (the reduction), 4x4 as 2x2 PEArrays (the stagger), 16x16 with three tiles
#: back to back (the schedule: K = M = 32 >= 2R - 1, so no bubbles)
STAGES = {
    1: Stage(1, "reduction", (2, 2), (1, 1), 3, 1, "value", True),
    2: Stage(2, "stagger", (2, 2), (2, 2), 4, 1, "index", False),
    3: Stage(3, "schedule", (16, 16), (1, 1), 32, 3, "front", False),
}


def stage_trace(stage: Stage, dataflow: int, *, seed: int = 0) -> Trace:
    """Record one stage of the progression from the RTL."""
    cfg = stage.config()
    A, B = operands(cfg, dataflow, stage.length, tiles=stage.tiles, small=stage.small, seed=seed)
    return record(cfg, dataflow, A, B)


def stage_svg(stage: Stage, dataflow: int) -> str:
    """The stage's animation; raises if the RTL departs from the predicted wavefront."""
    trace = stage_trace(stage, dataflow)
    problems = trace.check()
    if problems:
        raise AssertionError(
            f"stage {stage.number} ({'OS' if dataflow == OS else 'WS'}): "
            f"{len(problems)} departures from the predicted wavefront, first: {problems[0]}"
        )
    return render_svg(trace, view=stage.view)
