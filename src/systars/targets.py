"""
RTL generation targets: the single source of truth for what ``just gen``,
``just synth`` and CI generate and synthesize.

A *target* is a named, reproducible generation recipe: a component factory,
the Verilog top-level module name, and the output file (``<name>.v``). Port of
kpugen's ``kpugen.backends.verilog`` registry, restricted to the systolic
building blocks systars has.

Usage::

    python -m systars.targets list [--plain]       # '<name> <top>' per line with --plain
    python -m systars.targets gen [NAME ...] [--out-dir gen]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from amaranth.back import verilog
from amaranth.lib.wiring import Component

from .config import Dataflow, SystolicConfig


@dataclass(frozen=True)
class Target:
    """A named RTL generation recipe."""

    name: str
    """Target name on the command line, and the file name (``<name>.v``)."""

    top: str
    """Verilog top-level module name."""

    factory: Callable[[], Component]
    """Zero-argument callable that builds the Amaranth component."""

    description: str = ""

    @property
    def filename(self) -> str:
        return f"{self.name}.v"


def _pe() -> Component:
    from .core.pe import PE

    return PE(SystolicConfig())


def _pe_array(rows: int, cols: int) -> Callable[[], Component]:
    def build() -> Component:
        from .core.pe_array import PEArray

        return PEArray(SystolicConfig(tile_rows=rows, tile_cols=cols))

    return build


def _systolic_array(
    rows: int, cols: int, tile_rows: int = 1, tile_cols: int = 1
) -> Callable[[], Component]:
    def build() -> Component:
        from .core.systolic_array import SystolicArray

        return SystolicArray(
            SystolicConfig(grid_rows=rows, grid_cols=cols, tile_rows=tile_rows, tile_cols=tile_cols)
        )

    return build


def _scratchpad() -> Component:
    from .memory.scratchpad import Scratchpad

    return Scratchpad(SystolicConfig(sp_banks=2, sp_capacity_kb=4, spad_read_delay=2))


def _accumulator() -> Component:
    from .memory.accumulator import Accumulator

    return Accumulator(SystolicConfig(acc_banks=2, acc_capacity_kb=4, acc_latency=2))


def systolic_top_config() -> SystolicConfig:
    """The ``systolic_top`` target's configuration: a 4x4 array with every dataflow
    (OS, WS and IS, chosen per command by CONFIG), a 1 KiB scratchpad and a 2 KiB
    accumulator. The cocotb bench builds its ISA instructions from it."""
    return SystolicConfig(
        grid_rows=4,
        grid_cols=4,
        dataflow=Dataflow.OUTPUT_STATIONARY | Dataflow.B_STATIONARY | Dataflow.A_STATIONARY,
        sp_capacity_kb=1,
        sp_banks=4,
        acc_capacity_kb=2,
        acc_banks=2,
    )


def _systolic_top() -> Component:
    """The ``systolic_top`` target: ``SystolicTop`` on ``systolic_top_config()``."""
    from .top import SystolicTop

    return SystolicTop(systolic_top_config())


TARGETS: dict[str, Target] = {
    t.name: t
    for t in [
        Target("pe", "PE", _pe, "Systolic PE (default config)"),
        Target("pe_array", "PEArray", _pe_array(1, 1), "1x1 PEArray"),
        Target("pe_array_2x2", "PEArray_2x2", _pe_array(2, 2), "2x2 PEArray"),
        Target(
            "systolic_array",
            "SystolicArray",
            _systolic_array(2, 2, 2, 2),
            "2x2 grid of 2x2 PEArrays (4x4 PEs)",
        ),
        Target(
            "systolic_array_4x4", "SystolicArray_4x4", _systolic_array(4, 4), "4x4 SystolicArray"
        ),
        Target("scratchpad", "Scratchpad", _scratchpad, "2-bank 4 KiB scratchpad"),
        Target("accumulator", "Accumulator", _accumulator, "2-bank 4 KiB accumulator"),
        Target(
            "systolic_top",
            "SystolicTop",
            _systolic_top,
            "4x4 SystolicTop compute tile, OS/WS/IS (cocotb: Matmul, Conv2d)",
        ),
    ]
}


def emit_target(target: Target, out_dir: Path) -> Path:
    """Generate ``target`` into ``out_dir`` and return the written path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / target.filename
    path.write_text(verilog.convert(target.factory(), name=target.top), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point (``python -m systars.targets``)."""
    parser = argparse.ArgumentParser(prog="python -m systars.targets", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_list = sub.add_parser("list", help="List the RTL generation targets")
    p_list.add_argument("--plain", action="store_true", help="Print '<name> <top>' per line")
    p_gen = sub.add_parser("gen", help="Generate targets (default: all)")
    p_gen.add_argument("targets", nargs="*", help="Target names")
    p_gen.add_argument("--out-dir", type=Path, default=Path("gen"))
    args = parser.parse_args(argv)

    if args.command == "list":
        if args.plain:
            for t in TARGETS.values():
                print(t.name, t.top)
        else:
            width = max(len(n) for n in TARGETS)
            for t in TARGETS.values():
                print(f"{t.name:<{width}}  {t.top:<20} {t.description}")
        return 0

    unknown = [n for n in args.targets if n not in TARGETS]
    if unknown:
        print(f"error: unknown target(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(TARGETS)}", file=sys.stderr)
        return 2
    for t in [TARGETS[n] for n in args.targets] or TARGETS.values():
        print(f"Generated {emit_target(t, args.out_dir)} (top: {t.top})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
