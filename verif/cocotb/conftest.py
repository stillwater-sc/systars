"""
Systars Verification - Global pytest configuration and fixtures.

This module provides common fixtures and configuration for all cocotb tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project paths to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "verif" / "cocotb"))


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "verilator: marks tests requiring Verilator")
    config.addinivalue_line("markers", "ghdl: marks tests requiring GHDL")
    config.addinivalue_line("markers", "icarus: marks tests requiring Icarus Verilog")
    config.addinivalue_line("markers", "formal: marks formal verification tests")
    config.addinivalue_line("markers", "synthesis: marks tests requiring synthesis")


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Modify test collection based on available simulators."""
    sim = os.environ.get("SIM", "verilator").lower()

    skip_verilator = pytest.mark.skip(reason="Requires Verilator simulator")
    skip_ghdl = pytest.mark.skip(reason="Requires GHDL simulator")
    skip_icarus = pytest.mark.skip(reason="Requires Icarus Verilog simulator")

    for item in items:
        # Skip tests based on simulator availability
        if "verilator" in item.keywords and sim != "verilator":
            item.add_marker(skip_verilator)
        if "ghdl" in item.keywords and sim != "ghdl":
            item.add_marker(skip_ghdl)
        if "icarus" in item.keywords and sim != "icarus":
            item.add_marker(skip_icarus)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def src_dir(project_root) -> Path:
    """Return the source directory."""
    return project_root / "src"


@pytest.fixture(scope="session")
def gen_dir(project_root) -> Path:
    """Return the generated RTL directory."""
    return project_root / "gen"


@pytest.fixture(scope="session")
def verif_dir(project_root) -> Path:
    """Return the verification directory."""
    return project_root / "verif"


@pytest.fixture(scope="session")
def sim_name() -> str:
    """Return the current simulator name."""
    return os.environ.get("SIM", "verilator").lower()
