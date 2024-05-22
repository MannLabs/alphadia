conda create -n alphadia python=3.8 -y
conda activate alphadia
pip install -e '../.[stable,development-stable]'
alphadia -v
conda deactivate
