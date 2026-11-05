#!python

import warnings

__version__ = "2.4.0"

warnings.filterwarnings(
    "ignore", message="Dependency 'dask' not installed.", module="directlfq"
)
