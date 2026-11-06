"""Generate SystolicArray Verilog (gen/systolic_array.v, gen/systolic_array_4x4.v): the ``systolic_array`` targets of ``systars.targets``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from systars.targets import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(
        main(
            [
                "gen",
                "systolic_array",
                "systolic_array_4x4",
                "--out-dir",
                str(Path(__file__).parent.parent / "gen"),
            ]
        )
    )
