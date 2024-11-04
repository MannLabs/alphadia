set -e -u

INSTALL_TYPE=$1 # stable, loose, etc..
ENV_NAME=${2:-alphadia}
PYTHON_VERSION=${3:-3.11}

conda create -n $ENV_NAME python=$PYTHON_VERSION -y

if [ "$INSTALL_TYPE" = "loose" ]; then
  INSTALL_STRING=""
else
  INSTALL_STRING="[${INSTALL_TYPE}]"
fi

# print pip environment for reproducibility
conda run -n $ENV_NAME --no-capture-output pip freeze

# conda 'run' vs. 'activate', cf. https://stackoverflow.com/a/72395091
conda run -n $ENV_NAME --no-capture-output pip install -e "../.$INSTALL_STRING"
conda run -n $ENV_NAME --no-capture-output alphadia -v
