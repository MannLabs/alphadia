conda create -n alphadia python=3.9 -y
conda info --envs
conda init bash
conda activate alphadia
pip install -e '../.[development]'
alphadia -v
conda deactivate
