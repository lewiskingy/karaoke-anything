#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
. "$HOME/.cargo/env" 2>/dev/null || true

if ! command -v cargo-llvm-cov >/dev/null 2>&1; then
    echo "cargo-llvm-cov missing: rustup component add llvm-tools-preview && cargo install cargo-llvm-cov" >&2
    exit 1
fi

cd "$(dirname "$0")/../client"
# main.rs and device.rs are thin cpal/OS I/O glue -- talking to a real audio
# Host/Device, which can't be constructed in tests. The decision logic they
# call into is pulled out into device_selection.rs (and network.rs's
# AudioSocket-abstracted loops), which stay under the 100% requirement below.
cargo llvm-cov --ignore-filename-regex "main\.rs$|device\.rs$" --fail-under-lines 100
