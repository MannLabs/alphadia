#!python

import warnings

__version__ = "2.0.2"

warnings.filterwarnings(
    "ignore", message="Dependency 'dask' not installed.", module="directlfq"
)
