ENV_NAME=${1:-alphadia}

conda create -n $ENV_NAME python=3.9 -y

# conda 'run' vs. 'activate', cf. https://stackoverflow.com/a/72395091
conda run -n $ENV_NAME --no-capture-output pip install -e '../.[loose]'
conda run -n $ENV_NAME --no-capture-output alphadia -v
