ENV_NAME=${1:-alphadia}

conda create -n $ENV_NAME python=3.9 -y
#conda info --envs
#conda init bash
source /home/alphadia/miniconda3/etc/profile.d/conda.sh

conda activate $ENV_NAME
pip install -e '../.[development]'
alphadia -v
conda deactivate
