#!python

import warnings

__version__ = "2.3.1"

warnings.filterwarnings(
    "ignore", message="Dependency 'dask' not installed.", module="directlfq"
)
