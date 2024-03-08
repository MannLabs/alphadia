#!/bin/bash

# Initial cleanup
cd gui

rm -rf dist
rm -rf out

# Build the GUI using electron forge
npm run make