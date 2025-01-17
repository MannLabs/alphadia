#!/bin/bash
set -e -u

# Build the installer for MacOS.
# This script needs to be run from the root of the repository.
# Prerequisites: wheel has been build, e.g. using build_wheel.sh

rm -rf dist_pyinstaller build_pyinstaller

export EAGER_IMPORT=true  # TODO check if this can be removed with newest peptdeep version w/out transformer dependency

WHL_NAME=$(cd dist && ls ./*.whl && cd ..)
pip install "dist/${WHL_NAME}[stable]"

# Creating the stand-alone pyinstaller folder
pyinstaller release/pyinstaller/alphadia.spec --distpath dist_pyinstaller --workpath build_pyinstaller -y
