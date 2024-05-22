conda create -n alphadia python=3.9 -y
conda run -n alphadia --no-capture-output pip install -e '../.[development]'
conda run -n alphadia --no-capture-output alphadia -v
