#!/bin/bash
### Install the package from source with a given type in a defined conda environment with a define python version,
### and call it to check if it works
### example usage:
### ./pip_install.sh stable my_env 3.9
set -e -u

INSTALL_TYPE=$1 # stable, loose, etc..
ENV_NAME=${2:-alphadia}
PYTHON_VERSION=${3:-3.11}
INSTALL_MONO=${4:-false}
TORCH_VARIANT=${5:-GPU}

if [ "$(echo $INSTALL_MONO | tr '[:upper:]' '[:lower:]')" = "true" ]; then
  conda create -n $ENV_NAME python=$PYTHON_VERSION mono -y
else
  conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

if [ "$INSTALL_TYPE" = "loose" ]; then
  INSTALL_STRING=""
else
  INSTALL_STRING="[${INSTALL_TYPE}]"
fi

# print pip environment for reproducibility
conda run -n $ENV_NAME --no-capture-output pip freeze

# ability to install CPU-only PyTorch to save disk space (no CUDA packages)
# this is quite a hack as we assume a certain structure of the _requirements.freeze.txt file and modify it on the fly
# cf. also build_installer_linux.sh
# Note: will not work for "sys_platform == 'darwin' and platform_machine == 'x86_64'" (=MacOS w/Intel chip)
if [ "$(echo $TORCH_VARIANT | tr '[:upper:]' '[:lower:]')" = "cpu" ]; then
  echo "Installing CPU-only PyTorch f..."

  # Extract torch version for non-darwin platforms
  TORCH_VERSION=$(grep "torch==" ../requirements/_requirements.freeze.txt | grep "sys_platform != 'darwin' or platform_machine != 'x86_64'" | sed -E "s/torch==([0-9.]+).*/\1/")
  echo "Detected torch version: $TORCH_VERSION"

  cp ../requirements/_requirements.freeze.txt ../requirements/_requirements.freeze.txt.backup

  grep -v -E "^(torch==|nvidia-.*==|triton==)" ../requirements/_requirements.freeze.txt > ../requirements/_requirements.freeze.txt.tmp
  mv ../requirements/_requirements.freeze.txt.tmp ../requirements/_requirements.freeze.txt

  conda run -n $ENV_NAME --no-capture-output pip install --no-cache-dir "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu
fi

# Restore the original requirements file if it was backed up
if [ -f ../requirements/_requirements.freeze.txt.backup ]; then
  mv ../requirements/_requirements.freeze.txt.backup ../requirements/_requirements.freeze.txt
fi

# conda 'run' vs. 'activate', cf. https://stackoverflow.com/a/72395091
conda run -n $ENV_NAME --no-capture-output pip install -e "../.$INSTALL_STRING"

conda run -n $ENV_NAME --no-capture-output pip freeze

conda run -n $ENV_NAME --no-capture-output alphadia --check
