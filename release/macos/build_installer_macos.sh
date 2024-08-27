#!/bin/bash

# Build the installer for MacOS.
# This script needs to be run from the root of the repository.

pip install build
python -m build
pip install "dist/alphadia-1.7.2-py3-none-any.whl[stable]"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller
pyinstaller release/pyinstaller/alphadia.spec -y
