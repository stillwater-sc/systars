#!/usr/bin/env python3
"""
Energy Comparison: SIMT vs Systolic Array vs Stencil Machine.

This example demonstrates the energy efficiency advantage of dataflow
architectures (systolic arrays, stencil machines) over SIMT (GPU-style)
architectures for linear algebra workloads.

Key Insight:
    SIMT architectures spend ~90% of energy on instruction overhead:
    - Instruction fetch from I-cache
    - Instruction decode
    - Warp scheduling
    - Register file access (banked SRAM)
    - Operand collection

    Dataflow architectures (systolic, stencil) eliminate this overhead:
    - No instructions to fetch/decode
    - No warp scheduling
    - Data flows directly through compute units
    - Only pay for useful computation

Usage:
    python 02_energy_comparison.py [options]

Options:
    --size N        Matrix size for GEMM (default: 64)
    --channels C    Number of channels for Conv2D (default: 64)
    --kernel K      Kernel size for Conv2D (default: 3)
"""

import argparse
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit("/", 3)[0] + "/src")

from systars.simt import SIMTConfig
from systars.simt.energy_model import (
    Conv2DWorkload,
    GEMMWorkload,
    compare_architectures,
    estimate_simt_energy,
    estimate_stencil_energy,
    estimate_systolic_energy,
    print_comparison,
)


def format_energy(energy_pj: float) -> str:
    """Format energy value with appropriate unit."""
    if energy_pj >= 1e6:
        return f"{energy_pj / 1e6:.2f} µJ"
    elif energy_pj >= 1e3:
        return f"{energy_pj / 1e3:.2f} nJ"
    else:
        return f"{energy_pj:.1f} pJ"


def print_detailed_comparison(gemm: GEMMWorkload, conv2d: Conv2DWorkload):
    """Print detailed energy breakdown for each architecture."""
    config = SIMTConfig()

    print()
    print("=" * 80)
    print("DETAILED ENERGY BREAKDOWN")
    print("=" * 80)
    print()

    # GEMM on SIMT
    print("GEMM on SIMT (GPU-style):")
    print("-" * 40)
    simt_gemm = estimate_simt_energy(gemm, config)
    print(simt_gemm)
    print()

    # GEMM on Systolic
    print("GEMM on Systolic Array:")
    print("-" * 40)
    systolic_gemm = estimate_systolic_energy(gemm)
    print(systolic_gemm)
    print()

    # Conv2D on Stencil
    print("Conv2D on Stencil Machine:")
    print("-" * 40)
    stencil_conv = estimate_stencil_energy(conv2d)
    print(stencil_conv)
    print()


def print_bar_chart(title: str, items: list[tuple[str, float, str]]):
    """Print a simple ASCII bar chart."""
    print(f"\n{title}")
    print("-" * 60)

    max_val = max(val for _, val, _ in items)
    max_bar = 40

    for name, value, extra in items:
        bar_len = int((value / max_val) * max_bar)
        bar = "█" * bar_len
        print(f"{name:<12} {bar:<40} {extra}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Energy Comparison: SIMT vs Systolic vs Stencil")
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        help="Matrix size for GEMM (default: 64)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=64,
        help="Number of channels for Conv2D (default: 64)",
    )
    parser.add_argument(
        "--kernel",
        type=int,
        default=3,
        help="Kernel size for Conv2D (default: 3)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed energy breakdown",
    )

    args = parser.parse_args()

    # Create workloads
    gemm = GEMMWorkload(M=args.size, N=args.size, K=args.size)
    conv2d = Conv2DWorkload(
        input_height=args.size,
        input_width=args.size,
        input_channels=args.channels,
        output_channels=args.channels,
        kernel_size=args.kernel,
    )

    print()
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║    ENERGY EFFICIENCY COMPARISON: SIMT vs SYSTOLIC vs STENCIL            ║")
    print("║                                                                          ║")
    print("║    Demonstrating why dataflow architectures are more efficient          ║")
    print("║    for linear algebra than instruction-based SIMT architectures         ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()

    # Compare architectures
    results = compare_architectures(gemm=gemm, conv2d=conv2d)
    print_comparison(results)

    if args.detailed:
        print_detailed_comparison(gemm, conv2d)

    # Print efficiency chart
    config = SIMTConfig()
    simt_gemm = estimate_simt_energy(gemm, config)
    systolic_gemm = estimate_systolic_energy(gemm)
    stencil_conv = estimate_stencil_energy(conv2d)

    print_bar_chart(
        "Energy Efficiency (% energy spent on useful compute)",
        [
            ("SIMT", simt_gemm.efficiency_percent, f"{simt_gemm.efficiency_percent:.1f}%"),
            (
                "Systolic",
                systolic_gemm.efficiency_percent,
                f"{systolic_gemm.efficiency_percent:.1f}%",
            ),
            ("Stencil", stencil_conv.efficiency_percent, f"{stencil_conv.efficiency_percent:.1f}%"),
        ],
    )

    print_bar_chart(
        "Total Energy Consumption (lower is better)",
        [
            ("SIMT", simt_gemm.total_pj, format_energy(simt_gemm.total_pj)),
            ("Systolic", systolic_gemm.total_pj, format_energy(systolic_gemm.total_pj)),
            ("Stencil", stencil_conv.total_pj, format_energy(stencil_conv.total_pj)),
        ],
    )

    # Summary
    print()
    print("=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print()
    print("SIMT Architecture (GPU-style):")
    print("  - Every operation requires instruction fetch, decode, schedule")
    print("  - 3+ register file accesses per instruction (banked SRAM)")
    print("  - Bank conflicts cause stalls and wasted energy")
    print(f"  - Only {simt_gemm.efficiency_percent:.0f}% of energy goes to useful compute")
    print()
    print("Systolic Array:")
    print("  - No instructions - data flows through PE array")
    print("  - Local register access only (shift registers)")
    print("  - Perfect data reuse for GEMM (output-stationary dataflow)")
    print(f"  - {systolic_gemm.efficiency_percent:.0f}% of energy goes to useful compute")
    print(f"  - {results['gemm']['speedup']:.1f}× more energy efficient than SIMT for GEMM")
    print()
    print("Stencil Machine:")
    print("  - No instructions - streaming dataflow")
    print("  - Line buffers enable 1× DRAM read per input pixel")
    print("  - Perfect feature map reuse (vs 9× for im2col with 3×3 kernels)")
    print(f"  - {stencil_conv.efficiency_percent:.0f}% of energy goes to useful compute")
    print(f"  - {results['conv2d']['speedup']:.1f}× more energy efficient than SIMT for Conv2D")
    print()


if __name__ == "__main__":
    main()
