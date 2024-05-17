conda create -n alphadia python=3.9 -y
conda info --envs
conda init bash
source /home/alphadia/miniconda3/etc/profile.d/conda.sh

conda activate alphadia
pip install -e '../.[development]'
alphadia -v
conda deactivate
