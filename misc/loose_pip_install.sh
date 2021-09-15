conda create -n alphadia python=3.8 -y
conda activate alphadia
pip install -e '../.[development]'
alphadia
conda deactivate
