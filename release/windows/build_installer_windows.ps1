# Build the installer for Windows.
# This script must be run from the root of the repository.

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ./build
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ./dist

python -m build

# substitute X.Y.Z-devN with X.Y.Z.devN
$WHL_NAME = "alphadia-1.9.2-py3-none-any.whl" -replace '(\d+\.\d+\.\d+)-dev(\d+)', '$1.dev$2'
pip install "dist/$WHL_NAME[stable]"

# Creating the stand-alone pyinstaller folder
pip install tbb==2021.13.1
pyinstaller release/pyinstaller/alphadia.spec --distpath dist_pyinstaller --workpath build_pyinstaller -y
