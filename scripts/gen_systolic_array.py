#!/usr/bin/env python3
"""Generate SystolicArray Verilog from systars."""

import sys
from pathlib import Path

# Add src to path if running from scripts/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from amaranth.back import verilog  # noqa: E402

from systars.config import SystolicConfig  # noqa: E402
from systars.core.systolic_array import SystolicArray  # noqa: E402


def main():
    gen_dir = project_root / "gen"
    gen_dir.mkdir(exist_ok=True)

    # Generate 2x2 systolic array with 1x1 PEArrays (minimal with pipeline registers)
    config_2x2 = SystolicConfig(grid_rows=2, grid_cols=2, tile_rows=1, tile_cols=1)
    systolic_array_2x2 = SystolicArray(config_2x2)

    output_path = gen_dir / "systolic_array.v"
    with open(output_path, "w") as f:
        f.write(verilog.convert(systolic_array_2x2, name="SystolicArray"))

    print(f"Generated {output_path}")

    # Also generate a 4x4 systolic array for larger-scale testing
    config_4x4 = SystolicConfig(grid_rows=4, grid_cols=4, tile_rows=1, tile_cols=1)
    systolic_array_4x4 = SystolicArray(config_4x4)

    output_path_4x4 = gen_dir / "systolic_array_4x4.v"
    with open(output_path_4x4, "w") as f:
        f.write(verilog.convert(systolic_array_4x4, name="SystolicArray_4x4"))

    print(f"Generated {output_path_4x4}")


if __name__ == "__main__":
    main()
