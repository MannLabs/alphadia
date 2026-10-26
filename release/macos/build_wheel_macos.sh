#!/bin/bash
set -e -u

# Build the wheel used by the macos installer.
# This script must be run from the root of the repository.
# The wheel contains a compiled Rust extension and is therefore OS-specific: it is built natively on the
# installer runner (which ships a Rust toolchain), the alphashared workflow picks up this script automatically.
# The wheels published on PyPI are built separately by .github/workflows/_build_wheels.yml.

rm -rf dist ./*.egg-info

python -m build
