#!python

import warnings

__version__ = "2.0.0-dev0"

warnings.filterwarnings(
    "ignore", message="Dependency 'dask' not installed.", module="directlfq"
)
