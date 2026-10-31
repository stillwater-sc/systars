#!/usr/bin/env bash
#
# Install OSS CAD Suite (Verilator, GHDL, Yosys, etc.)
# https://github.com/YosysHQ/oss-cad-suite-build
#
# Installs into $OSS_CAD_SUITE (default /opt/oss-cad-suite).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/oss-cad-suite.sh
source "${SCRIPT_DIR}/lib/oss-cad-suite.sh"

OSS_CAD_SUITE_VERSION="2025-12-12"
OSS_CAD_SUITE_DIR="${OSS_CAD_SUITE:-/opt/oss-cad-suite}"

install_oss_cad_suite "$OSS_CAD_SUITE_VERSION" "$OSS_CAD_SUITE_DIR"

echo ""
echo "Add to your shell configuration:"
echo "    export PATH=\"${OSS_CAD_SUITE_DIR}/bin:\$PATH\""
