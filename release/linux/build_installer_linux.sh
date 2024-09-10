#!/bin/bash
set -e -u

# Build the installer for Linux.
# This script must be run from the root of the repository.

rm -rf dist build *.egg-info
rm -rf dist_pyinstaller build_pyinstaller

# Creating the wheel
python setup.py sdist bdist_wheel

# Setting up the local package
# Make sure you include the required extra packages and always use the stable or very-stable options!
pip install "dist/alphadia-1.7.2-py3-none-any.whl[stable]"

# Creating the stand-alone pyinstaller folder
pyinstaller release/pyinstaller/alphadia.spec --distpath dist_pyinstaller --workpath build_pyinstaller -y
