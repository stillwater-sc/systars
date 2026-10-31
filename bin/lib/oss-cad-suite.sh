# shellcheck shell=bash
#
# Shared OSS CAD Suite installer, sourced by install-oss-cad-suite.sh,
# upgrade-oss-cad-suite.sh and setup-dev-env.sh.
#
#   install_oss_cad_suite VERSION PREFIX
#
# 1. Downloads the release archive into a private directory (mktemp -d, mode
#    0700, owned by the invoking user), never a predictable /tmp path.
# 2. Verifies the archive (gzip CRC + tar listing) before anything privileged
#    reads it. YosysHQ does not publish checksums for these archives, so this
#    detects truncated or corrupt downloads, not a compromised release.
# 3. Extracts into a staging directory next to PREFIX and checks that it holds
#    a working layout (bin/yosys).
# 4. Only then replaces PREFIX. The previous installation is kept until the
#    swap succeeds and is restored if it fails.

OSS_CAD_SUITE_REPO="YosysHQ/oss-cad-suite-build"

oss_cad_suite_url() {
    local version="$1"
    local date_compact="${version//-/}"
    echo "https://github.com/${OSS_CAD_SUITE_REPO}/releases/download/${version}/oss-cad-suite-linux-x64-${date_compact}.tgz"
}

# Runs in a subshell so its EXIT trap and `set -e` cannot leak into the caller.
install_oss_cad_suite() (
    set -euo pipefail
    local version="$1" prefix="$2"
    local url workdir archive staging backup
    url="$(oss_cad_suite_url "$version")"
    workdir="$(mktemp -d)"
    archive="${workdir}/oss-cad-suite.tgz"
    staging="${prefix}.staging.$$"
    backup="${prefix}.previous.$$"

    cleanup() {
        rm -rf "$workdir"
        if [[ -e "$staging" ]]; then sudo rm -rf "$staging"; fi
    }
    trap cleanup EXIT

    echo "Downloading OSS CAD Suite ${version}..." >&2
    echo "  ${url}" >&2
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$archive" "$url"
    elif command -v curl &>/dev/null; then
        curl -fL --proto '=https' --tlsv1.2 --progress-bar -o "$archive" "$url"
    else
        echo "Error: neither wget nor curl is available" >&2
        exit 1
    fi

    echo "Verifying archive..." >&2
    if ! gzip -t "$archive" || ! tar -tzf "$archive" >/dev/null; then
        echo "Error: downloaded archive is corrupt or truncated" >&2
        exit 1
    fi

    echo "Extracting into staging directory ${staging}..." >&2
    sudo mkdir -p "$staging"
    sudo tar -xzf "$archive" -C "$staging"
    if [[ ! -x "${staging}/oss-cad-suite/bin/yosys" ]]; then
        echo "Error: archive does not contain oss-cad-suite/bin/yosys" >&2
        exit 1
    fi

    if [[ -e "$prefix" ]]; then
        sudo mv "$prefix" "$backup"
    fi
    if ! sudo mv "${staging}/oss-cad-suite" "$prefix"; then
        echo "Error: could not install into ${prefix}; restoring previous installation" >&2
        if [[ -e "$backup" ]]; then sudo mv "$backup" "$prefix"; fi
        exit 1
    fi
    if [[ -e "$backup" ]]; then sudo rm -rf "$backup"; fi
    echo "OSS CAD Suite ${version} installed at ${prefix}" >&2
)
