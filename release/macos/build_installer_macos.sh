#!/bin/bash
set -e -u

# Build the installer for MacOS.
# This script needs to be run from the root of the repository.

rm -rf dist build *.egg-info
rm -rf dist_pyinstaller build_pyinstaller

export EAGER_IMPORT=true  # TODO check if this can be removed with newset peptdeep version w/out transformer dependenc

python -m build
pip install "dist/alphadia-1.9.2-py3-none-any.whl[stable]"

# Creating the stand-alone pyinstaller folder
pyinstaller release/pyinstaller/alphadia.spec --distpath dist_pyinstaller --workpath build_pyinstaller -y
