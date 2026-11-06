#!/usr/bin/env python3
"""
RTL Wavefront Animation.

Runs the real SystolicArray RTL in Amaranth simulation, records what every PE
holds in every clock, checks it against the predicted wavefront, and draws it.
Unlike 02/03 (a Python model of a textbook array), this is the generated
hardware: every PE registers its inputs, so the wavefront advances one PE per
clock (driven as the L1 stream buffers drive it, each lane one clock after the
last), tiles follow each other on the Streamer's schedule, and in
output-stationary mode the results drain north.

The stages of the docs' progression (docs/wavefront-animation.md):

    --stage 1   2x2, the reduction: operands and running sums (value view)
    --stage 2   4x4 as 2x2 PEArrays, the stagger: which k / m meets where (index view)
    --stage 3   16x16, three tiles back to back, the schedule (front view)

Any option given after --stage overrides the stage's setting.

Usage:
    python 08_rtl_wavefront.py [--stage 1|2|3] [--grid R C] [--tile R C]
                               [--dataflow os|ws] [--n N] [--tiles T]
                               [--view auto|value|index|front] [--small]
                               [--svg out.svg] [--text]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from systars.config import SystolicConfig
from systars.wavefront import OS, STAGES, VIEWS, WS, operands, record, render_svg, render_text


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", type=int, choices=sorted(STAGES), help="A stage's settings")
    parser.add_argument("--grid", type=int, nargs=2, metavar=("R", "C"), help="PEArray grid")
    parser.add_argument("--tile", type=int, nargs=2, metavar=("R", "C"), help="PEs per PEArray")
    parser.add_argument("--dataflow", choices=["os", "ws"], default="os")
    parser.add_argument("--n", type=int, help="K (OS) or M, the streamed rows (WS), per tile")
    parser.add_argument("--tiles", type=int, help="Tiles back to back (the Streamer's schedule)")
    parser.add_argument("--view", choices=["auto", *VIEWS], help="SVG view (default: by size)")
    parser.add_argument("--small", action="store_true", help="Single-digit operands")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--svg", type=Path, help="Write an animated SVG")
    parser.add_argument("--text", action="store_true", help="Print the frames as text")
    args = parser.parse_args()

    stage = STAGES[args.stage] if args.stage else STAGES[2]
    grid = args.grid or list(stage.grid)
    tile = args.tile or list(stage.tile)
    n = stage.length if args.n is None else args.n
    tiles = args.tiles or (stage.tiles if args.stage else 1)
    view = args.view or (stage.view if args.stage else "auto")
    small = args.small or (stage.small if args.stage else False)

    cfg = SystolicConfig(grid_rows=grid[0], grid_cols=grid[1], tile_rows=tile[0], tile_cols=tile[1])
    dataflow = OS if args.dataflow == "os" else WS
    A, B = operands(cfg, dataflow, n, tiles=tiles, small=small, seed=args.seed)
    trace = record(cfg, dataflow, A, B)

    if args.text:
        print(render_text(trace))
    if args.svg:
        svg = render_svg(trace, view=view)
        args.svg.write_text(svg, encoding="utf-8")
        print(f"Wrote {args.svg} ({len(svg) // 1024} KB)")

    problems = trace.check()
    if problems:
        print(f"{len(problems)} departures from the predicted wavefront:")
        for p in problems[:20]:
            print(f"  {p}")
        sys.exit(1)
    checked = sum(1 for f in trace.frames if f.phase == "stream")
    starts = ", ".join(str(t.start) for t in trace.tiles)
    print(
        f"{trace.rows}x{trace.cols} PEs, {len(trace.tiles)} tile(s) starting at clock(s) "
        f"{starts}: the RTL matches the predicted wavefront in all {checked} stream clocks, "
        f"every result left the array on time, and C == A @ B."
    )


if __name__ == "__main__":
    main()
