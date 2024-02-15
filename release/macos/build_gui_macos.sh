#!/bin/bash

# Cleanup the GUI build
rm -rf gui/dist
rm -rf gui/out

# Build the GUI using electron forge
npm run make --prefix gui