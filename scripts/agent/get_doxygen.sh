#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Install CI's pinned Doxygen once, into ~/.local/opt (no root), and print its path.
# The version and sha256 are read from .github/workflows/ci.yml so the two cannot drift.
#
#   DOXYGEN=$(scripts/agent/get_doxygen.sh)
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
ver=$(sed -n 's/^ *DOXYGEN_VERSION: *//p' "$root/.github/workflows/ci.yml" | head -n 1)
sha=$(sed -n 's/^ *DOXYGEN_SHA256: *//p' "$root/.github/workflows/ci.yml" | head -n 1)
[ -n "$ver" ] && [ -n "$sha" ] || { echo "get_doxygen: no DOXYGEN_VERSION/SHA256 in ci.yml" >&2; exit 1; }
dest=${BATCHLAS_DOXYGEN_PREFIX:-$HOME/.local/opt}
bin=$dest/doxygen-$ver/bin/doxygen
if [ ! -x "$bin" ]; then
    mkdir -p "$dest"
    tgz=$dest/doxygen-$ver.linux.bin.tar.gz
    tag="Release_$(echo "$ver" | tr . _)"
    echo "get_doxygen: downloading Doxygen $ver" >&2
    curl -fsSL -o "$tgz" "https://github.com/doxygen/doxygen/releases/download/$tag/doxygen-$ver.linux.bin.tar.gz"
    echo "$sha  $tgz" | sha256sum -c - >&2
    tar -xzf "$tgz" -C "$dest"
    rm -f "$tgz"
fi
"$bin" --version >/dev/null
echo "$bin"
