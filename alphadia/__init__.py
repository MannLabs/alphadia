#!python

import warnings

__version__ = "2.0.3"

warnings.filterwarnings(
    "ignore", message="Dependency 'dask' not installed.", module="directlfq"
)
