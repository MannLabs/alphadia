# navigate to the root directory

# Creating the wheel
# Creating the wheel
python setup.py sdist bdist_wheel
pip install "dist/alphadia-1.5.4-py3-none-any.whl[stable]"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller tbb
pyinstaller release/pyinstaller/alphadia.spec -y