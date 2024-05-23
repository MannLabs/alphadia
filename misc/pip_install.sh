set -e -u

INSTALL_TYPE=$1 # stable, loose, etc..
ENV_NAME=${2:-alphadia}

conda create -n $ENV_NAME python=3.9 -y

# conda 'run' vs. 'activate', cf. https://stackoverflow.com/a/72395091
conda run -n $ENV_NAME --no-capture-output pip install -e "../.[${INSTALL_TYPE}]"
conda run -n $ENV_NAME --no-capture-output alphadia -v
