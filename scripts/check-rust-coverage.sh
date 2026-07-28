#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
. "$HOME/.cargo/env" 2>/dev/null || true

if ! command -v cargo-llvm-cov >/dev/null 2>&1; then
    echo "cargo-llvm-cov missing: rustup component add llvm-tools-preview && cargo install cargo-llvm-cov" >&2
    exit 1
fi

cd "$(dirname "$0")/../client"
cargo llvm-cov --ignore-filename-regex "main\.rs$" --fail-under-lines 100
