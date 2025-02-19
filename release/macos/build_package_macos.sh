#!/bin/bash
set -e -u

# Build the install package for MacOS.
# This script must be run from the root of the repository after running build_installer_macos.sh and build_gui_macos.sh

# Set up package name and version
PACKAGE_NAME="alphadia"
APP_NAME="alphadia"
PACKAGE_VERSION="1.9.4-dev0"
PKG_FOLDER="dist/$APP_NAME.app"

# BUILD_NAME is taken from environment variables, e.g. alphadia-1.9.4-dev0-macos-darwin-arm64 or alphadia-1.9.4-dev0-macos-darwin-x64
rm -rf ${BUILD_NAME}.pkg

# Cleanup the package folder
echo "Cleaning up the package folder"
rm -rf $PKG_FOLDER

# === Prepare structure ===
echo "Preparing package structure"
mkdir -p $PKG_FOLDER/Contents/Resources
mkdir -p $PKG_FOLDER/Contents/MacOS
mkdir -p $PKG_FOLDER/Contents/Frameworks

# === Backend ===
echo "Copying backend"
BACKEND_BUILD="dist_pyinstaller/$PACKAGE_NAME"

if [ ! -d "$BACKEND_BUILD" ]; then
  echo "Backend build not found at $BACKEND_BUILD"
  exit 1
fi

cp -a ${BACKEND_BUILD}/. $PKG_FOLDER/Contents/Frameworks/


# === GUI ===
echo "Copying GUI"

# TODO find out where this contract from ARCH & KERNEL is used in GUI building
ARCH=$(uname -m)
if [ "$ARCH" == "x86_64" ]; then
  ARCH="x64"
fi
KERNEL=$(uname -s | tr '[:upper:]' '[:lower:]')
echo ARCH=${ARCH} KERNEL=${KERNEL}

ls ./gui/out

GUI_BUILD="./gui/out/${PACKAGE_NAME}-gui-${KERNEL}-${ARCH}"
if [ ! -d "$GUI_BUILD" ]; then
  echo "GUI build not found at $GUI_BUILD"
  exit 1
fi

cp -a ${GUI_BUILD}/. $PKG_FOLDER/Contents/Frameworks/

# === Resources ===
echo "Copying resources"
CONTENTS_FOLDER=$PKG_FOLDER/Contents
cp release/logos/$PACKAGE_NAME.icns ${CONTENTS_FOLDER}/Resources
cp release/logos/$PACKAGE_NAME.png ${CONTENTS_FOLDER}/Resources
cp release/macos/$APP_NAME ${CONTENTS_FOLDER}/MacOS

cp release/macos/Info.plist ${CONTENTS_FOLDER}

# change permissions for entry script
chmod +x ${CONTENTS_FOLDER}/MacOS/$APP_NAME

pkgbuild --root $PKG_FOLDER --identifier de.mpg.biochem.$PACKAGE_NAME.app --version $PACKAGE_VERSION --install-location /Applications/$APP_NAME.app --scripts release/macos/scripts $BUILD_NAME.pkg --nopayload
