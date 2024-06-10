set -e -u

INSTALL_TYPE=$1 # stable, loose, etc..
ENV_NAME=${2:-alphadia}

conda create -n $ENV_NAME python=3.9 -y

if [ "$INSTALL_TYPE" = "loose" ]; then
  INSTALL_STRING=""
else
  INSTALL_STRING="[${INSTALL_TYPE}]"
fi

# conda 'run' vs. 'activate', cf. https://stackoverflow.com/a/72395091
conda run -n $ENV_NAME --no-capture-output pip install --no-cache-dir -e "../.$INSTALL_STRING"
conda run -n $ENV_NAME --no-capture-output alphadia -v
