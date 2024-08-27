#!/bin/bash

# Build the GUI for MacOS.
# This script needs to be run from the root of the repository.

# Cleanup the GUI build
rm -rf gui/dist
rm -rf gui/out

npm install --prefix gui
# Build the GUI using electron forge
npm run make --prefix gui
