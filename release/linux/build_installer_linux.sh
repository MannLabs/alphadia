#!/bin/bash
set -e -u

CPU_OR_GPU=${1:-CPU}

# Build the installer for Linux.
# This script must be run from the root of the repository.
# Prerequisites: wheel has been build, e.g. using build_wheel.sh

rm -rf dist_pyinstaller build_pyinstaller


WHL_NAME=$(cd dist && ls ./*.whl && cd ..)
pip install "dist/${WHL_NAME}[stable]"

if [ "${CPU_OR_GPU}" != "GPU" ]; then
    pip install torch -U --extra-index-url https://download.pytorch.org/whl/cpu
fi

# Creating the stand-alone pyinstaller folder
pyinstaller release/pyinstaller/alphadia.spec --distpath dist_pyinstaller --workpath build_pyinstaller -y

# to avoid 'no space left on device' error
rm -r build_pyinstaller
