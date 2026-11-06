"""Generate PEArray Verilog (gen/pe_array.v, gen/pe_array_2x2.v): the ``pe_array`` targets of ``systars.targets``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from systars.targets import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(
        main(
            [
                "gen",
                "pe_array",
                "pe_array_2x2",
                "--out-dir",
                str(Path(__file__).parent.parent / "gen"),
            ]
        )
    )
