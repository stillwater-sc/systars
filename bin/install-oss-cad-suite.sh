#!/usr/bin/env bash
#
# Install OSS CAD Suite (Verilator, GHDL, Yosys, etc.)
# https://github.com/YosysHQ/oss-cad-suite-build

set -euo pipefail

OSS_CAD_SUITE_VERSION="2025-12-12"
OSS_CAD_SUITE_DATE=$(echo "$OSS_CAD_SUITE_VERSION" | tr -d '-')
OSS_CAD_TARBALL="oss-cad-suite-linux-x64-${OSS_CAD_SUITE_DATE}.tgz"
OSS_CAD_URL="https://github.com/YosysHQ/oss-cad-suite-build/releases/download/${OSS_CAD_SUITE_VERSION}/${OSS_CAD_TARBALL}"

echo "Downloading OSS CAD Suite ${OSS_CAD_SUITE_VERSION}..."
wget -q --show-progress -O "/tmp/${OSS_CAD_TARBALL}" "$OSS_CAD_URL"

echo "Extracting to /opt..."
sudo tar -xzf "/tmp/${OSS_CAD_TARBALL}" -C /opt
rm "/tmp/${OSS_CAD_TARBALL}"

echo ""
echo "OSS CAD Suite installed to /opt/oss-cad-suite"
echo ""
echo "Add to your shell configuration:"
echo '    export PATH="/opt/oss-cad-suite/bin:$PATH"'
