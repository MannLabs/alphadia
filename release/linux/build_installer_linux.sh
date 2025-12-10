#!/bin/bash
set -e -u

# CPU is default as this is what we need for releasing
CPU_OR_GPU=${1:-CPU}

# Build the installer for Linux.
# This script must be run from the root of the repository.
# Prerequisites: wheel has been build, e.g. using build_wheel.sh

rm -rf dist_pyinstaller build_pyinstaller


WHL_NAME=$(cd dist && ls ./*.whl && cd ..)
pip install "dist/${WHL_NAME}[stable]"

# TODO: remove when GitHub allows for release artifacts > 2 GB
if [ "${CPU_OR_GPU}" == "CPU" ]; then
    # this is a bit flaky and depends on the format of the requirements freeze file, cf. also pip_install.sh
    TORCH_VERSION=$(grep "torch==" requirements/_requirements.freeze.txt | grep "sys_platform != 'darwin' or platform_machine != 'x86_64'" | sed -E "s/torch==([0-9.]+).*/\1/")
    echo "Detected torch version: $TORCH_VERSION"
    pip install torch==$TORCH_VERSION -U --extra-index-url https://download.pytorch.org/whl/cpu
fi

# Creating the stand-alone pyinstaller folder
pyinstaller release/pyinstaller/alphadia.spec --distpath dist_pyinstaller --workpath build_pyinstaller -y

# to avoid 'no space left on device' error
rm -r build_pyinstaller
