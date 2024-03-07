#!/bin/bash

# Cleanup the GUI build
rm -rf gui/dist
rm -rf gui/out

npm install --prefix gui
# Build the GUI using electron forge
npm run make --prefix gui