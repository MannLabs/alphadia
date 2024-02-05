#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

# Initial cleanup
conda remove -n alphadiainstaller --all -y

# navigate to the root directory
cd ../..

# Creating a conda environment
conda create -n alphadiainstaller python=3.9 -y
conda activate alphadiainstaller

# Creating the wheel
# Creating the wheel
python setup.py sdist bdist_wheel
pip install "dist/alphadia-1.5.3-py3-none-any.whl[stable]"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller
pyinstaller release/pyinstaller/alphadia.spec -y
conda deactivate