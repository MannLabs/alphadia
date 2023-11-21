#!bash

# Initial cleanup
rm -rf dist
rm -rf build
FILE=AlphaDIA.pkg
if test -f "$FILE"; then
  rm AlphaDIA.pkg
fi
cd ../..
rm -rf dist
rm -rf build

# Creating a conda environment
conda create -n alphadiainstaller python=3.8 -y
conda activate alphadiainstaller

# Creating the wheel
python setup.py sdist bdist_wheel

# Setting up the local package
cd release/one_click_macos_gui
pip install "../../dist/alphadia-1.4.0-py3-none-any.whl[stable]"

# Creating the stand-alone pyinstaller folder
pip install pyinstaller==4.2
pyinstaller ../pyinstaller/alphadia.spec -y
conda deactivate

# If needed, include additional source such as e.g.:
# cp ../../alphadia/data/*.fasta dist/alphadia/data

# Wrapping the pyinstaller folder in a .pkg package
mkdir -p dist/alphadia/Contents/Resources
cp ../logos/alpha_logo.icns dist/alphadia/Contents/Resources
mv dist/alphadia_gui dist/alphadia/Contents/MacOS
cp Info.plist dist/alphadia/Contents
cp alphadia_terminal dist/alphadia/Contents/MacOS
cp ../../LICENSE.txt Resources/LICENSE.txt
cp ../logos/alpha_logo.png Resources/alpha_logo.png
chmod 777 scripts/*

pkgbuild --root dist/alphadia --identifier de.mpg.biochem.alphadia.app --version 0.3.0 --install-location /Applications/AlphaDIA.app --scripts scripts AlphaDIA.pkg
productbuild --distribution distribution.xml --resources Resources --package-path AlphaDIA.pkg dist/alphadia_gui_installer_macos.pkg
