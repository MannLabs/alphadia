#!bash

# Initial cleanup
rm -rf dist
rm -rf build
cd ../..
rm -rf dist
rm -rf build

# Creating a conda environment
conda create -n alphadia_installer python=3.8 -y
conda activate alphadia_installer

# Creating the wheel
pip install build
python -m build

# Setting up the local package
cd release/one_click_linux_gui
# Make sure you include the required extra packages and always use the stable or very-stable options!
pip install "../../dist/alphadia-1.5.5+test-py3-none-any.whl"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller==4.2
pyinstaller ../pyinstaller/alphadia.spec -y
conda deactivate

# If needed, include additional source such as e.g.:
# cp ../../alphadia/data/*.fasta dist/alphadia/data
# WARNING: this probably does not work!!!!

# Wrapping the pyinstaller folder in a .deb package
mkdir -p dist/AlphaDIA_gui_installer_linux/usr/local/bin
mv dist/AlphaDIA dist/AlphaDIA_gui_installer_linux/usr/local/bin/AlphaDIA
mkdir dist/AlphaDIA_gui_installer_linux/DEBIAN
cp control dist/AlphaDIA_gui_installer_linux/DEBIAN
dpkg-deb --build --root-owner-group dist/AlphaDIA_gui_installer_linux/
