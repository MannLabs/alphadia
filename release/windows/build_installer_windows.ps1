# Build the installer for Windows.
# This script must be run from the root of the repository.
# Prerequisites: the wheel for this platform is in dist/

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ./build_pyinstaller
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ./dist_pyinstaller


$WHL_NAME = (Get-ChildItem -Path "dist" -Filter "*.whl").Name
pip install "dist/$WHL_NAME[stable]"

# Creating the stand-alone pyinstaller folder
# pip install tbb==2021.13.1  # TODO check if this is necessary
pyinstaller release/pyinstaller/alphadia.spec --distpath dist_pyinstaller --workpath build_pyinstaller -y
