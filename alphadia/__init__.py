#!python

import warnings

__version__ = "1.11.0"

warnings.filterwarnings(
    "ignore", message="Dependency 'dask' not installed.", module="directlfq"
)
