"""
Common CLI argument definitions for animation examples.

This module provides shared argument groups that can be added to argparse
parsers in animation scripts, ensuring consistent CLI interfaces across
all visualization tools.

Usage:
    from examples.common.cli import add_animation_args, add_timeline_args

    parser = argparse.ArgumentParser()
    add_animation_args(parser)  # Adds --delay, --fast, --step, --movie, etc.
    add_timeline_args(parser)   # Adds --timeline, --timeline-format
    args = parser.parse_args()

    # Use the unified accessor
    delay = args.effective_delay  # Returns 0 if --fast, else args.delay
"""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass


@dataclass
class AnimationArgs:
    """
    Parsed animation arguments with computed properties.

    This provides a cleaner interface than accessing raw argparse Namespace.
    """

    delay: int
    fast: bool
    step: bool
    movie: bool
    no_color: bool
    max_cycles: int

    @property
    def effective_delay(self) -> int:
        """Get effective delay (0 if fast mode enabled)."""
        return 0 if self.fast else self.delay

    @classmethod
    def from_namespace(cls, args: Namespace) -> "AnimationArgs":
        """Create from argparse Namespace with defaults for missing attrs."""
        return cls(
            delay=getattr(args, "delay", 500),
            fast=getattr(args, "fast", False),
            step=getattr(args, "step", False),
            movie=getattr(args, "movie", False),
            no_color=getattr(args, "no_color", False),
            max_cycles=getattr(args, "max_cycles", 1000),
        )


def add_animation_args(
    parser: ArgumentParser,
    *,
    default_delay: int = 500,
    default_max_cycles: int = 1000,
    include_fast: bool = True,
    include_movie: bool = True,
    include_no_color: bool = True,
    include_max_cycles: bool = True,
) -> None:
    """
    Add common animation control arguments to a parser.

    Args:
        parser: ArgumentParser to add arguments to
        default_delay: Default delay in milliseconds (default: 500)
        default_max_cycles: Default max cycles (default: 1000)
        include_fast: Include --fast flag (default: True)
        include_movie: Include --movie flag (default: True)
        include_no_color: Include --no-color flag (default: True)
        include_max_cycles: Include --max-cycles flag (default: True)

    Adds these arguments:
        --delay MS      Delay between frames in milliseconds
        --fast          Fast mode (no animation delay)
        --step          Step mode (press Enter to advance each cycle)
        --movie         Movie mode: suppress setup/summary for term2svg
        --no-color      Disable colored output
        --max-cycles N  Maximum cycles to run before stopping
    """
    group = parser.add_argument_group("Animation Control")

    group.add_argument(
        "--delay",
        type=int,
        default=default_delay,
        metavar="MS",
        help=f"Delay between frames in milliseconds (default: {default_delay})",
    )

    if include_fast:
        group.add_argument(
            "--fast",
            action="store_true",
            help="Fast mode (no animation delay)",
        )

    group.add_argument(
        "--step",
        action="store_true",
        help="Step mode (press Enter to advance each cycle)",
    )

    if include_movie:
        group.add_argument(
            "--movie",
            action="store_true",
            help="Movie mode: suppress prompts and setup/summary for term2svg capture",
        )

    if include_no_color:
        group.add_argument(
            "--no-color",
            action="store_true",
            help="Disable colored output",
        )

    if include_max_cycles:
        group.add_argument(
            "--max-cycles",
            type=int,
            default=default_max_cycles,
            metavar="N",
            help=f"Maximum cycles to run before stopping (default: {default_max_cycles})",
        )


def add_timeline_args(parser: ArgumentParser) -> None:
    """
    Add timeline logging arguments to a parser.

    Adds these arguments:
        --timeline FILE         Enable timeline logging to file
        --timeline-format FMT   Timeline format: csv or chrome (default: chrome)
    """
    group = parser.add_argument_group("Timeline Logging")

    group.add_argument(
        "--timeline",
        type=str,
        default=None,
        metavar="FILE",
        help="Enable timeline logging to file (e.g., trace.json or events.csv)",
    )

    group.add_argument(
        "--timeline-format",
        type=str,
        default="chrome",
        choices=["csv", "chrome"],
        help="Timeline format: csv or chrome (default: chrome)",
    )


def add_gemm_args(
    parser: ArgumentParser,
    *,
    default_m: int = 8,
    default_n: int = 8,
    default_k: int = 4,
) -> None:
    """
    Add GEMM dimension arguments to a parser.

    Args:
        parser: ArgumentParser to add arguments to
        default_m: Default M dimension (default: 8)
        default_n: Default N dimension (default: 8)
        default_k: Default K dimension (default: 4)

    Adds these arguments:
        --m M   M dimension for GEMM output rows
        --n N   N dimension for GEMM output cols
        --k K   K dimension for GEMM reduction
    """
    group = parser.add_argument_group("GEMM Dimensions")

    group.add_argument(
        "--m",
        type=int,
        default=default_m,
        help=f"M dimension for GEMM output rows (default: {default_m})",
    )

    group.add_argument(
        "--n",
        type=int,
        default=default_n,
        help=f"N dimension for GEMM output cols (default: {default_n})",
    )

    group.add_argument(
        "--k",
        type=int,
        default=default_k,
        help=f"K dimension for GEMM reduction (default: {default_k})",
    )


def add_memory_args(
    parser: ArgumentParser,
    *,
    default_latency: int = 200,
    include_fast_mem: bool = True,
) -> None:
    """
    Add memory configuration arguments to a parser.

    Args:
        parser: ArgumentParser to add arguments to
        default_latency: Default memory latency in cycles (default: 200)
        include_fast_mem: Include --fast-mem flag (default: True)

    Adds these arguments:
        --fast-mem              Use fast memory latencies (1 cycle)
        --mem-latency CYCLES    Memory latency in cycles
    """
    group = parser.add_argument_group("Memory Configuration")

    if include_fast_mem:
        group.add_argument(
            "--fast-mem",
            action="store_true",
            help="Use fast memory latencies for demo (1 cycle instead of default)",
        )

    group.add_argument(
        "--mem-latency",
        type=int,
        default=None,
        metavar="CYCLES",
        help=f"Memory latency in cycles (default: {default_latency}, overrides --fast-mem)",
    )


def get_effective_delay(args: Namespace) -> int:
    """
    Get the effective animation delay from parsed args.

    Returns 0 if --fast is set, otherwise returns args.delay.
    """
    if getattr(args, "fast", False):
        return 0
    return getattr(args, "delay", 200)


def should_print(args: Namespace) -> bool:
    """
    Check if setup/summary output should be printed.

    Returns False if --movie mode is enabled.
    """
    return not getattr(args, "movie", False)


def should_prompt(args: Namespace) -> bool:
    """
    Check if interactive prompts should be shown.

    Returns False if --movie mode is enabled (prompts break term2svg capture).
    """
    return not getattr(args, "movie", False)
