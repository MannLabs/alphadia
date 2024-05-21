conda create -n alphadia python=3.9 -y
conda run -n pip install -e '../.[development]'
alphadia -v
