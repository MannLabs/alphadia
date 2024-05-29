#!/bin/bash
# extract the current version from the code and print, e.g. 1.5.5

grep "__version__" alphadia/__init__.py | cut -f3 -d ' ' | sed 's/"//g'
