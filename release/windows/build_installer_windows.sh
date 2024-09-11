# Build the installer for Windows.
# This script must be run from the root of the repository.

rm -rf dist build *.egg-info
rm -rf dist_pyinstaller build_pyinstaller

python -m build
pip install "dist/alphadia-1.7.2-py3-none-any.whl[stable]"

# Creating the stand-alone pyinstaller folder
pip install tbb==2021.13.1
pyinstaller release/pyinstaller/alphadia.spec --distpath dist_pyinstaller --workpath build_pyinstaller -y
