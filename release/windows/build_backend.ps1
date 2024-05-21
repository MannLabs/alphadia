Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ./build
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ./dist

pip install git+https://github.com/huggingface/huggingface_hub.git
python -c 'from huggingface_hub import get_full_repo_name; print("success")'

pip install build
python -m build
pip install "dist/alphadia-1.5.5+test-py3-none-any.whl"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller tbb
pyinstaller release/pyinstaller/alphadia.spec -y
