conda create -n alphadia python=3.8 -y
conda activate alphadia
pip install -e '../.'
alphadia -v
conda deactivate
