cd ../..
conda create -n alphadia_pypi_wheel python=3.8
conda activate alphadia_pypi_wheel
pip install twine
rm -rf dist
rm -rf build
pip install build
python -m build
twine check dist/*
conda deactivate
