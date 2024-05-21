conda create -n alphadia python=3.9 -y
conda run -n alphadia pip install -e '../.[development]'
conda run -n alphadia alphadia -v
