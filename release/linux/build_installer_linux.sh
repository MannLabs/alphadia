#!/bin/bash
set -e -u

CPU_OR_GPU=${1:-CPU}

# Build the installer for Linux.
# This script must be run from the root of the repository.

rm -rf dist build *.egg-info
rm -rf dist_pyinstaller build_pyinstaller

python -m build
# substitute X.Y.Z-devN with X.Y.Z.devN
WHL_NAME=$(echo "alphadia-1.9.2-py3-none-any.whl" | sed 's/\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)-dev\([0-9][0-9]*\)/\1.dev\2/g')
pip install "dist/${WHL_NAME}[stable]"

if [ "${CPU_OR_GPU}" != "GPU" ]; then
    pip install torch -U --extra-index-url https://download.pytorch.org/whl/cpu
fi

# Creating the stand-alone pyinstaller folder
pyinstaller release/pyinstaller/alphadia.spec --distpath dist_pyinstaller --workpath build_pyinstaller -y

# to avoid 'no space left on device' error
rm -r build_pyinstaller
