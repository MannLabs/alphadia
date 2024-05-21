# navigate to the root directory

# Creating the wheel
pip install git+https://github.com/huggingface/huggingface_hub.git
python -c "from huggingface_hub import get_full_repo_name; print('success')"

pip install build
python -m build
pip install "dist/alphadia-1.5.5-py3-none-any.whl[stable]"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller
pyinstaller release/pyinstaller/alphadia.spec -y
