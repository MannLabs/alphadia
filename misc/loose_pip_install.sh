conda create -n alphadia python=3.9 -y
conda activate alphadia
pip install -e '../.[development]'
alphadia -v
conda deactivate