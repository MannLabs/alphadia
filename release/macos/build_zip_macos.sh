#!/bin/bash

# Set up package name and version
PACKAGE_NAME="alphadia"
PACKAGE_VERSION="1.5.3"

ARCH=$(uname -m)
if [ "$ARCH" == "x86_64" ]; then
  ARCH="amd64"
fi
KERNEL=$(uname -s | tr '[:upper:]' '[:lower:]')

BUILD_NAME="${PACKAGE_NAME}-${PACKAGE_VERSION}-${KERNEL}-${ARCH}"

# Cleanup the dist folder
rm -rf ../../dist/${BUILD_NAME}

# Create the dist folder
mkdir -p ../../dist/${BUILD_NAME}

# Copy the electron forge build
cp -a ../../gui/out/alphadia-gui-${KERNEL}-${ARCH}/. ../../dist/${BUILD_NAME}

# Copy the backend
cp -a ../../dist/alphadia/. ../../dist/${BUILD_NAME}

# create the zip file
cd ../../dist
zip -r ${BUILD_NAME}.zip ${BUILD_NAME}