#!/usr/bin/env python3
"""
RTL Wavefront Animation.

Runs the real SystolicArray RTL in Amaranth simulation, records what every PE
sees in every cycle, checks it against the predicted wavefront, and draws it.
Unlike 02/03 (a Python model of a textbook array), this is the generated
hardware: inside a PEArray data passes combinationally, so the wavefront
advances one PEArray per cycle.

Each cell shows which element is meeting at that PE, read back from the RTL's
own values: ``kN`` (output-stationary: A[i, N] meets B[N, j]) or ``mN``
(weight-stationary: row N of A). A cell where A and B disagree shows ``X``.

Usage:
    python 08_rtl_wavefront.py [--grid R C] [--tile R C] [--dataflow os|ws] [--n N]
                               [--svg out.svg] [--quiet]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from systars.config import SystolicConfig
from systars.wavefront import OS, WS, label_operands, record, render_svg, render_text


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--grid", type=int, nargs=2, default=[2, 2], metavar=("R", "C"), help="PEArray grid"
    )
    parser.add_argument(
        "--tile", type=int, nargs=2, default=[2, 2], metavar=("R", "C"), help="PEs per PEArray"
    )
    parser.add_argument("--dataflow", choices=["os", "ws"], default="os")
    parser.add_argument("--n", type=int, default=4, help="K (OS) or rows of A (WS)")
    parser.add_argument("--svg", type=Path, help="Write an animated SVG")
    parser.add_argument("-q", "--quiet", action="store_true", help="Do not print the frames")
    args = parser.parse_args()

    cfg = SystolicConfig(
        grid_rows=args.grid[0],
        grid_cols=args.grid[1],
        tile_rows=args.tile[0],
        tile_cols=args.tile[1],
    )
    dataflow = OS if args.dataflow == "os" else WS
    A, B = label_operands(cfg, dataflow, args.n)
    trace = record(cfg, dataflow, A, B)

    if not args.quiet:
        print(render_text(trace))
    if args.svg:
        args.svg.write_text(render_svg(trace))
        print(f"Wrote {args.svg}")

    problems = trace.check()
    if problems:
        print(f"{len(problems)} departures from the predicted wavefront:")
        for p in problems[:20]:
            print(f"  {p}")
        sys.exit(1)
    print(f"RTL matches the predicted wavefront in all {len(trace.frames)} cycles, and C == A @ B.")


if __name__ == "__main__":
    main()
