#!/bin/bash

# Set up package name and version
PACKAGE_NAME="alphadia"
PACKAGE_VERSION="1.5.3"

BUILD_NAME="${PACKAGE_NAME}-${PACKAGE_VERSION}-win-x64"

# Cleanup the dist folder
rm -rf dist/${BUILD_NAME}

# Create the dist folder
mkdir -p dist/${BUILD_NAME}

# === GUI ===

#ls gui/out

# Set the path to the GUI build
#GUI_BUILD="gui/out/alphadia-gui-${KERNEL}-${ARCH}"

# Check if the GUI build exists, otherwise exit with an error
#if [ ! -d "$GUI_BUILD" ]; then
#  echo "GUI build not found at $GUI_BUILD"
#  exit 1
#fi

# Copy the electron forge build
#cp -a ${GUI_BUILD}/. dist/${BUILD_NAME}

# === Backend ===

BACKEND_BUILD="dist/alphadia"

# Check if the backend build exists, otherwise exit with an error
if [ ! -d "$BACKEND_BUILD" ]; then
  echo "Backend build not found at $BACKEND_BUILD"
  exit 1
fi

# Copy the backend
cp -a ${BACKEND_BUILD}/. dist/${BUILD_NAME}

# create the zip file
cd dist
powershell Compress-Archive ${BUILD_NAME} ${BUILD_NAME}.zip