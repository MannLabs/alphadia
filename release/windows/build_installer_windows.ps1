# Build the installer for Windows.
# This script must be run from the root of the repository.

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ./build
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ./dist

pip install build
python -m build
pip install "dist/alphadia-1.7.2-py3-none-any.whl[stable]"

# Creating the stand-alone pyinstaller folder
pip install tbb==2021.13.1
pyinstaller release/pyinstaller/alphadia.spec --distpath dist_pyinstaller --workpath build_pyinstaller -y
