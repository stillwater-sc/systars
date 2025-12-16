#!/usr/bin/env python3
"""Generate Mesh Verilog from systars."""

import sys
from pathlib import Path

# Add src to path if running from scripts/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from amaranth.back import verilog  # noqa: E402

from systars.config import SystolicConfig  # noqa: E402
from systars.core.mesh import Mesh  # noqa: E402


def main():
    gen_dir = project_root / "gen"
    gen_dir.mkdir(exist_ok=True)

    # Generate 2x2 mesh with 1x1 tiles (minimal mesh with pipeline registers)
    config_2x2 = SystolicConfig(mesh_rows=2, mesh_cols=2, tile_rows=1, tile_cols=1)
    mesh_2x2 = Mesh(config_2x2)

    output_path = gen_dir / "mesh.v"
    with open(output_path, "w") as f:
        f.write(verilog.convert(mesh_2x2, name="Mesh"))

    print(f"Generated {output_path}")

    # Also generate a 4x4 mesh for larger-scale testing
    config_4x4 = SystolicConfig(mesh_rows=4, mesh_cols=4, tile_rows=1, tile_cols=1)
    mesh_4x4 = Mesh(config_4x4)

    output_path_4x4 = gen_dir / "mesh_4x4.v"
    with open(output_path_4x4, "w") as f:
        f.write(verilog.convert(mesh_4x4, name="Mesh_4x4"))

    print(f"Generated {output_path_4x4}")


if __name__ == "__main__":
    main()
