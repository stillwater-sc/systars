#!/usr/bin/env python3
"""
Generate the wavefront animations the docs embed (docs/wavefront/).

Records every stage of the progression (systars.wavefront.STAGES: 2x2 the
reduction, 4x4 the stagger, 16x16 the schedule) from the RTL, OS and WS, checks
each against the predicted wavefront, and writes the SVGs. The SVGs are
committed; ``--check`` regenerates them in memory and fails if a committed file
is missing or differs (tests/unit/test_wavefront.py checks the same in CI).

Usage:
    python scripts/gen_wavefronts.py [--check] [--out DIR]
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from systars.wavefront import OS, STAGES, WS, stage_svg  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="Fail if a committed SVG is stale")
    parser.add_argument("--out", type=Path, default=project_root / "docs" / "wavefront")
    args = parser.parse_args()

    stale = []
    args.out.mkdir(parents=True, exist_ok=True)
    for stage in STAGES.values():
        for dataflow in (OS, WS):
            path = args.out / stage.filename(dataflow)
            svg = stage_svg(stage, dataflow)  # raises if the RTL departs from the prediction
            if args.check:
                if not path.exists() or path.read_text() != svg:
                    stale.append(path)
                continue
            path.write_text(svg)
            print(f"Wrote {path} ({len(svg) // 1024} KB)")
    if stale:
        for path in stale:
            print(f"stale: {path}")
        print("Run `just wavefronts` and commit the result.")
        sys.exit(1)
    if args.check:
        print(f"All {2 * len(STAGES)} wavefront animations are current.")


if __name__ == "__main__":
    main()
