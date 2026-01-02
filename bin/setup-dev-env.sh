#!/usr/bin/env bash
# Systars Development Environment Setup Script
# Run this script on a fresh Ubuntu/Debian machine to set up the build environment
#
# Usage: ./bin/setup-dev-env.sh [--all|--minimal|--help]
#
# Options:
#   --minimal   Install only essential dependencies (no optional tools)
#   --all       Install everything including optional tools (default)
#   --help      Show this help message

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# OSS CAD Suite version
OSS_CAD_SUITE_VERSION="2025-12-12"
OSS_CAD_SUITE_DATE=$(echo "$OSS_CAD_SUITE_VERSION" | tr -d '-')

# Logging functions
info() { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# Check if command exists
has_cmd() { command -v "$1" &>/dev/null; }

# Show help
show_help() {
    head -15 "$0" | tail -12
    exit 0
}

# Parse arguments
INSTALL_MODE="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal) INSTALL_MODE="minimal"; shift ;;
        --all) INSTALL_MODE="all"; shift ;;
        --help|-h) show_help ;;
        *) error "Unknown option: $1" ;;
    esac
done

echo ""
echo "================================================"
echo "  Systars Development Environment Setup"
echo "================================================"
echo ""
info "Install mode: $INSTALL_MODE"
info "Project root: $PROJECT_ROOT"
echo ""

# Check OS
if [[ ! -f /etc/debian_version ]]; then
    warn "This script is designed for Ubuntu/Debian. Some commands may not work."
fi

# ============================================================================
# System Packages
# ============================================================================
info "Installing system packages..."

sudo apt-get update

# Essential packages
ESSENTIAL_PKGS=(
    build-essential
    git
    curl
    wget
    python3
    python3-pip
    python3-venv
)

# Optional packages
OPTIONAL_PKGS=(
    graphviz
    gtkwave
)

sudo apt-get install -y "${ESSENTIAL_PKGS[@]}"
success "Essential packages installed"

if [[ "$INSTALL_MODE" == "all" ]]; then
    sudo apt-get install -y "${OPTIONAL_PKGS[@]}" || warn "Some optional packages failed to install"
    success "Optional packages installed"
fi

# ============================================================================
# Just (task runner)
# ============================================================================
if has_cmd just; then
    success "Just already installed: $(just --version)"
else
    info "Installing Just task runner..."
    curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
    success "Just installed: $(just --version)"
fi

# ============================================================================
# OSS CAD Suite (Verilator, GHDL, Yosys)
# ============================================================================
OSS_CAD_SUITE_PATH="/opt/oss-cad-suite"

if [[ -d "$OSS_CAD_SUITE_PATH" ]]; then
    success "OSS CAD Suite already installed at $OSS_CAD_SUITE_PATH"
else
    info "Installing OSS CAD Suite (Verilator, GHDL, Yosys)..."

    OSS_CAD_TARBALL="oss-cad-suite-linux-x64-${OSS_CAD_SUITE_DATE}.tgz"
    OSS_CAD_URL="https://github.com/YosysHQ/oss-cad-suite-build/releases/download/${OSS_CAD_SUITE_VERSION}/${OSS_CAD_TARBALL}"

    info "Downloading from: $OSS_CAD_URL"
    wget -q --show-progress -O "/tmp/${OSS_CAD_TARBALL}" "$OSS_CAD_URL"

    info "Extracting to /opt..."
    sudo tar -xzf "/tmp/${OSS_CAD_TARBALL}" -C /opt
    rm "/tmp/${OSS_CAD_TARBALL}"

    success "OSS CAD Suite installed"
fi

# Add OSS CAD Suite bin to PATH for tools (yosys, verilator, etc.)
export PATH="$OSS_CAD_SUITE_PATH/bin:$PATH"
success "OSS CAD Suite tools added to PATH"

# ============================================================================
# Python Virtual Environment and Dependencies
# ============================================================================
info "Setting up Python virtual environment..."

cd "$PROJECT_ROOT"

if [[ ! -d .venv ]]; then
    python3 -m venv .venv
    success "Created virtual environment"
else
    success "Virtual environment already exists"
fi

# Activate venv
source .venv/bin/activate

info "Upgrading pip..."
pip install --upgrade pip

info "Installing Python dependencies..."
pip install -e ".[all]"
success "Python dependencies installed"

# ============================================================================
# Pre-commit Hooks
# ============================================================================
info "Installing pre-commit hooks..."
pre-commit install
success "Pre-commit hooks installed"

# ============================================================================
# Patch venv activate script to include OSS CAD Suite
# ============================================================================
info "Patching venv activation script to include OSS CAD Suite..."
ACTIVATE_SCRIPT="$PROJECT_ROOT/.venv/bin/activate"
if ! grep -q "oss-cad-suite" "$ACTIVATE_SCRIPT"; then
    sed -i 's|PATH="\$VIRTUAL_ENV/"bin":\$PATH"|PATH="$VIRTUAL_ENV/"bin":$PATH"\n# Add OSS CAD Suite to PATH if available\nif [ -d "/opt/oss-cad-suite/bin" ]; then\n    PATH="/opt/oss-cad-suite/bin:$PATH"\nfi|' "$ACTIVATE_SCRIPT"
    success "Patched activate script"
else
    success "Activate script already patched"
fi

# ============================================================================
# Verification
# ============================================================================
echo ""
echo "================================================"
echo "  Verifying Installation"
echo "================================================"
echo ""

verify_tool() {
    local name=$1
    local cmd=$2
    local version
    if has_cmd "$cmd"; then
        version=$($cmd --version 2>&1 | head -1)
        success "$name: $version"
        return 0
    else
        warn "$name: NOT FOUND"
        return 1
    fi
}

verify_tool "Python" python3
verify_tool "Pip" pip
verify_tool "Just" just
verify_tool "Verilator" verilator
verify_tool "GHDL" ghdl
verify_tool "Yosys" yosys
verify_tool "Cocotb" cocotb-config
verify_tool "Ruff" ruff
verify_tool "Pre-commit" pre-commit

echo ""
echo "================================================"
echo "  Setup Complete!"
echo "================================================"
echo ""
info "To activate the environment in a new shell:"
echo ""
echo "    cd $PROJECT_ROOT"
echo "    source .venv/bin/activate"
echo ""
info "Common commands:"
echo ""
echo "    just              # Show available tasks"
echo "    just test         # Run tests"
echo "    just lint         # Run linters"
echo "    just gen          # Generate Verilog"
echo ""
success "Happy development!"
