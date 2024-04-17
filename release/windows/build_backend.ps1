Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ./build
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ./dist

python setup.py sdist bdist_wheel
pip install "dist/alphadia-1.5.5-py3-none-any.whl[stable]"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller tbb
pyinstaller release/pyinstaller/alphadia.spec -y