#!/usr/bin/env bash
#
# Upgrade OSS CAD Suite to a new version
# https://github.com/YosysHQ/oss-cad-suite-build
#
# Usage:
#   ./upgrade-oss-cad-suite.sh <version>
#   ./upgrade-oss-cad-suite.sh 2025-12-15
#   ./upgrade-oss-cad-suite.sh --list    # Show recent releases
#
# The version should be in YYYY-MM-DD format (e.g., 2025-12-12)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/oss-cad-suite.sh
source "${SCRIPT_DIR}/lib/oss-cad-suite.sh"

OSS_CAD_SUITE_DIR="${OSS_CAD_SUITE:-/opt/oss-cad-suite}"
GITHUB_REPO="$OSS_CAD_SUITE_REPO"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 <version>"
    echo ""
    echo "Arguments:"
    echo "  version    OSS CAD Suite version in YYYY-MM-DD format (e.g., 2025-12-12)"
    echo "  --list     Show recent releases from GitHub"
    echo "  --current  Show currently installed version"
    echo ""
    echo "Examples:"
    echo "  $0 2025-12-15"
    echo "  $0 --list"
    exit 1
}

show_current_version() {
    if [[ -d "$OSS_CAD_SUITE_DIR" ]]; then
        if [[ -f "$OSS_CAD_SUITE_DIR/VERSION" ]]; then
            echo "Current version: $(cat "$OSS_CAD_SUITE_DIR/VERSION")"
        elif [[ -x "$OSS_CAD_SUITE_DIR/bin/yosys" ]]; then
            echo "Current version: $("$OSS_CAD_SUITE_DIR/bin/yosys" --version 2>/dev/null | head -1 || echo "unknown")"
        else
            echo "OSS CAD Suite installed at $OSS_CAD_SUITE_DIR (version unknown)"
        fi
    else
        echo "OSS CAD Suite not installed at $OSS_CAD_SUITE_DIR"
    fi
}

list_releases() {
    echo "Fetching recent OSS CAD Suite releases..."
    echo ""
    if command -v gh &> /dev/null; then
        gh release list --repo "$GITHUB_REPO" --limit 10
    elif command -v curl &> /dev/null; then
        curl -s "https://api.github.com/repos/${GITHUB_REPO}/releases?per_page=10" | \
            grep '"tag_name"' | \
            sed 's/.*"tag_name": "\([^"]*\)".*/\1/' | \
            head -10
    else
        echo "Install 'gh' CLI or 'curl' to list releases"
        echo "Or visit: https://github.com/${GITHUB_REPO}/releases"
        exit 1
    fi
}

validate_version() {
    local version="$1"
    # Check format YYYY-MM-DD
    if [[ ! "$version" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        echo -e "${RED}Error: Invalid version format '$version'${NC}"
        echo "Version must be in YYYY-MM-DD format (e.g., 2025-12-12)"
        exit 1
    fi
}

check_release_exists() {
    local version="$1"
    local url
    url="$(oss_cad_suite_url "$version")"

    echo "Checking if release $version exists..."
    if curl --output /dev/null --silent --head --fail "$url"; then
        echo -e "${GREEN}Release $version found${NC}"
        return 0
    else
        echo -e "${RED}Error: Release $version not found${NC}"
        echo "Check available releases with: $0 --list"
        exit 1
    fi
}

verify_installation() {
    echo ""
    echo "Verifying installation..."

    if [[ ! -d "$OSS_CAD_SUITE_DIR" ]]; then
        echo -e "${RED}Error: Installation directory not found${NC}"
        exit 1
    fi

    local tools=("yosys" "verilator" "iverilog")
    local ver

    for tool in "${tools[@]}"; do
        if [[ -x "$OSS_CAD_SUITE_DIR/bin/$tool" ]]; then
            ver=$("$OSS_CAD_SUITE_DIR/bin/$tool" --version 2>/dev/null | head -1 || echo "ok")
            echo -e "  ${GREEN}✓${NC} $tool: $ver"
        else
            echo -e "  ${YELLOW}?${NC} $tool: not found (may be optional)"
        fi
    done

    echo ""
    echo -e "${GREEN}OSS CAD Suite upgraded successfully!${NC}"
    echo ""
    echo "Make sure your PATH includes:"
    echo "    export PATH=\"${OSS_CAD_SUITE_DIR}/bin:\$PATH\""
}

# Main
if [[ $# -eq 0 ]]; then
    usage
fi

case "$1" in
    --list|-l)
        list_releases
        exit 0
        ;;
    --current|-c)
        show_current_version
        exit 0
        ;;
    --help|-h)
        usage
        ;;
    *)
        TARGET_VERSION="$1"
        ;;
esac

# Validate version format
validate_version "$TARGET_VERSION"

# Show current version
echo "=== OSS CAD Suite Upgrade ==="
echo ""
show_current_version
echo "Target version: $TARGET_VERSION"

# Check if release exists
check_release_exists "$TARGET_VERSION"

# Confirm with user
echo ""
read -p "Proceed with upgrade? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Download, verify and stage the new version; the existing installation is
# replaced only after the new one is extracted and checked.
install_oss_cad_suite "$TARGET_VERSION" "$OSS_CAD_SUITE_DIR"

# Verify
verify_installation
