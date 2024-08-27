# TODO remove with old release workflow

# navigate to the root directory

pip install build
python -m build
pip install "dist/alphadia-1.7.2-py3-none-any.whl[stable]"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller
pyinstaller release/pyinstaller/alphadia.spec -y
