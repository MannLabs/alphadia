ENV_NAME=${1:-alphadia}

conda create -n $ENV_NAME python=3.9 -y

# conda 'run' vs. 'activate', cf. https://stackoverflow.com/a/72395091
conda run -n $ENV_NAME --live-stream pip install -e '../.[development]'
conda run -n $ENV_NAME alphadia -v
