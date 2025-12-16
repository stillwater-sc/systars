#!/usr/bin/env python3
"""Generate Tile Verilog from systars."""

import sys
from pathlib import Path

# Add src to path if running from scripts/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from amaranth.back import verilog  # noqa: E402

from systars.config import SystolicConfig  # noqa: E402
from systars.core.tile import Tile  # noqa: E402


def main():
    gen_dir = project_root / "gen"
    gen_dir.mkdir(exist_ok=True)

    # Generate default 1x1 tile
    config_1x1 = SystolicConfig(tile_rows=1, tile_cols=1)
    tile_1x1 = Tile(config_1x1)

    output_path = gen_dir / "tile.v"
    with open(output_path, "w") as f:
        f.write(verilog.convert(tile_1x1, name="Tile"))

    print(f"Generated {output_path}")

    # Also generate a 2x2 tile for testing
    config_2x2 = SystolicConfig(tile_rows=2, tile_cols=2)
    tile_2x2 = Tile(config_2x2)

    output_path_2x2 = gen_dir / "tile_2x2.v"
    with open(output_path_2x2, "w") as f:
        f.write(verilog.convert(tile_2x2, name="Tile_2x2"))

    print(f"Generated {output_path_2x2}")


if __name__ == "__main__":
    main()
