#!/bin/bash
set -e -u

# Build the install package for Linux.
# This script must be run from the root of the repository after running build_installer_linux.sh

PACKAGE_NAME=alphadia

# BUILD_NAME is taken from environment variables, e.g. 'alphadia-1.2.1-linux-x64'
rm -rf ${BUILD_NAME}.deb

BIN_PATH=dist_pyinstaller/${BUILD_NAME}/usr/local/${PACKAGE_NAME}
mkdir -p ${BIN_PATH}
chmod 755 ${BIN_PATH}

# === GUI ===
echo "Copying GUI"

# TODO find out where this contract from ARCH & KERNEL is used in GUI building
ARCH=$(uname -m)
if [ "$ARCH" == "x86_64" ]; then
  ARCH="x64"
fi
KERNEL=$(uname -s | tr '[:upper:]' '[:lower:]')
echo ARCH=${ARCH} KERNEL=${KERNEL}

echo ls ./gui/out
ls ./gui/out
echo

GUI_BUILD="./gui/out/${PACKAGE_NAME}-gui-${KERNEL}-${ARCH}"
if [ ! -d "$GUI_BUILD" ]; then
  echo "GUI build not found at $GUI_BUILD"
  exit 1
fi
echo ls ${GUI_BUILD}
ls ${GUI_BUILD}
echo

cp -a ${GUI_BUILD}/. ${BIN_PATH}


mkdir -p dist_pyinstaller/${BUILD_NAME}/usr/local/bin
cd dist_pyinstaller/${BUILD_NAME}/usr/local/bin
ln -s ../alphadia/alphadia alphadia
ln -s ../alphadia/alphadia-gui alphadia-gui
cd -

# Wrapping the pyinstaller folder in a .deb package
mv dist_pyinstaller/${PACKAGE_NAME} ${BIN_PATH}
mkdir dist_pyinstaller/${BUILD_NAME}/DEBIAN
cp release/linux/control dist_pyinstaller/${BUILD_NAME}/DEBIAN

dpkg-deb --build --root-owner-group dist_pyinstaller/${BUILD_NAME}

# release workflow expects artifact at root of repository
mv dist_pyinstaller/${BUILD_NAME}.deb .
