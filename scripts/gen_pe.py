#!/usr/bin/env python3
"""Generate PE Verilog from systars."""

import sys
from pathlib import Path

# Add src to path if running from scripts/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from amaranth.back import verilog  # noqa: E402

from systars.config import SystolicConfig  # noqa: E402
from systars.core.pe import PE  # noqa: E402


def main():
    gen_dir = project_root / "gen"
    gen_dir.mkdir(exist_ok=True)

    config = SystolicConfig()
    pe = PE(config)

    output_path = gen_dir / "pe.v"
    with open(output_path, "w") as f:
        f.write(verilog.convert(pe, name="PE"))

    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
