#!/bin/bash

# Set up package name and version
PACKAGE_NAME="alphadia"
PACKAGE_VERSION="1.5.5"

ARCH=$(uname -m)
if [ "$ARCH" == "x86_64" ]; then
  ARCH="x64"
fi
echo "ARCH=${ARCH}" >> $GITHUB_ENV

KERNEL=$(uname -s | tr '[:upper:]' '[:lower:]')

BUILD_NAME="${PACKAGE_NAME}-${PACKAGE_VERSION}-${KERNEL}-${ARCH}"

PKG_FOLDER="dist/alphaDIA.app"

# Cleanup the package folder
echo "Cleaning up the package folder"

rm -rf $PKG_FOLDER

# === Prepare structure ===
echo "Preparing package structure"

mkdir -p $PKG_FOLDER
mkdir -p $PKG_FOLDER/Contents/Resources
mkdir -p $PKG_FOLDER/Contents/MacOS
mkdir -p $PKG_FOLDER/Contents/Frameworks



# === Backend ===
echo "Copying backend"

BACKEND_BUILD="dist/alphadia"

# Check if the backend build exists, otherwise exit with an error
if [ ! -d "$BACKEND_BUILD" ]; then
  echo "Backend build not found at $BACKEND_BUILD"
  exit 1
fi

# Copy the backend
cp -a ${BACKEND_BUILD}/. $PKG_FOLDER/Contents/Frameworks/


# === GUI ===
echo "Copying GUI"

ls ./gui/out

# Set the path to the GUI build
GUI_BUILD="./gui/out/alphadia-gui-${KERNEL}-${ARCH}"

# Check if the GUI build exists, otherwise exit with an error
if [ ! -d "$GUI_BUILD" ]; then
  echo "GUI build not found at $GUI_BUILD"
  exit 1
fi

# Copy the electron forge build
cp -a ${GUI_BUILD}/. $PKG_FOLDER/Contents/Frameworks/

# === Resources ===
echo "Copying resources"
cp release/logos/alphadia.icns $PKG_FOLDER/Contents/Resources/
cp release/logos/alphadia.png $PKG_FOLDER/Contents/Resources/
cp release/macos/alphaDIA $PKG_FOLDER/Contents/MacOS/

cp release/macos/Info.plist $PKG_FOLDER/Contents/

#change permissions for entry script
chmod +x $PKG_FOLDER/Contents/MacOS/alphaDIA

pkgbuild --root $PKG_FOLDER --identifier de.mpg.biochem.alphadia.app --version $PACKAGE_VERSION --install-location /Applications/alphaDIA.app --scripts release/macos/scripts dist/$BUILD_NAME.pkg --nopayload