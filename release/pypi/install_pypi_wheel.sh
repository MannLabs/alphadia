conda create -n alphadia_pip_test python=3.8 -y
conda activate alphadia_pip_test
pip install "alphadia[stable]"
alphadia
conda deactivate
