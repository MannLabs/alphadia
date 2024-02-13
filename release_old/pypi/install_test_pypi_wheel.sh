conda create -n alphadia_pip_test python=3.8 -y
conda activate alphadia_pip_test
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "alphadia[stable]"
alphadia
conda deactivate
