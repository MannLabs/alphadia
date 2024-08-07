Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ./build
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ./dist

pip install build
python -m build
pip install "dist/alphadia-1.7.2-py3-none-any.whl[stable]"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller tbb
pyinstaller release/pyinstaller/alphadia.spec -y
