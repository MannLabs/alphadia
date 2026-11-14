#!python

import warnings

__version__ = "2.5.0"

warnings.filterwarnings(
    "ignore", message="Dependency 'dask' not installed.", module="directlfq"
)
