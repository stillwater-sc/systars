"""``python -m systars.trace BUNDLE [--check] [--svg DIR] [--perf]``: report on a recorded run.

kpu-gen's ``kpu-gen trace`` command (branes-ai/embodied-ai-soc-generator), for a systars
tile's bundle (#18). Prints the text report; with ``--check`` also the trace checks (exit 1
on a problem); with ``--svg DIR`` writes the SVG views; ``--perf`` prints only the
performance summary, as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .check import check_trace
from .perf import summary
from .record import Trace
from .render import text_report, write_all


def _err(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Print a trace bundle's report; optionally check it and draw its SVGs."""
    parser = argparse.ArgumentParser(
        prog="python -m systars.trace", description="Report on a recorded run (a trace bundle)"
    )
    parser.add_argument("bundle", type=Path, metavar="BUNDLE")
    parser.add_argument("--check", action="store_true", help="run the trace checks")
    parser.add_argument(
        "--allow-errors", action="store_true", help="error responses are not problems"
    )
    parser.add_argument("--svg", type=Path, metavar="DIR", help="write the SVG views to DIR")
    parser.add_argument("--rows", type=int, default=40, help="rows listed (40)")
    parser.add_argument(
        "--perf", action="store_true", help="print the run's performance summary (JSON) only"
    )
    args = parser.parse_args(argv)

    if args.perf and (args.check or args.svg is not None):
        _err("--perf prints only the summary: it does not combine with --check or --svg")
        return 1
    try:
        trace = Trace.load(args.bundle)
    except (OSError, ValueError, KeyError) as exc:
        _err(f"{args.bundle}: {exc}")
        return 1
    if args.perf:  # the performance summary alone, as JSON (perf.py)
        print(json.dumps(summary(trace), indent=2))
        return 0
    print(text_report(trace, max_rows=args.rows), end="")
    status = 0
    if args.check:
        problems = check_trace(trace, allow_errors=args.allow_errors)
        print(f"\ntrace checks: {'clean' if not problems else f'{len(problems)} problems'}")
        for problem in problems:
            print(f"  {problem}")
        status = 1 if problems else 0
    if args.svg is not None:
        # run.1.json and run.2.json stay apart; seed-0.trace.json -> seed-0
        stem = args.bundle.name.removesuffix(".json").removesuffix(".trace")
        for path in write_all(trace, args.svg, stem):
            print(f"wrote {path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
