#!/bin/bash
set -e -u

# Build the installer for MacOS.
# This script needs to be run from the root of the repository.

python -m build
pip install "dist/alphadia-1.7.2-py3-none-any.whl[stable]"

# Creating the stand-alone pyinstaller folder
pyinstaller release/pyinstaller/alphadia.spec --distpath dist_pyinstaller --workpath build_pyinstaller -y
